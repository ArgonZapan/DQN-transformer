import math
import torch
import torch.nn as nn
from .grn   import GRN
from .vsn   import VSN
from .heads import QuantileHead, ThresholdHead


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TFBlock(nn.Module):
    """Single TFT attention block: MHA + GRN post-attention."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.grn  = GRN(d_model, ff_dim, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.grn(x)


class PerVariableProjection(nn.Module):
    """Per-variable scalar → d_model embedding: each feature has its own w_f, b_f.

    Replaces a single shared Linear(1, d_model), under which every feature's
    embedding was colinear (the same direction scaled by the feature value) and
    only the downstream VSN GRNs could tell variables apart. The TFT paper
    prescribes one linear projection per variable — this restores that.
    """

    def __init__(self, n_vars: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_vars, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_vars, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_vars] → [batch, seq_len, n_vars, d_model]
        return x.unsqueeze(-1) * self.weight + self.bias


class TimeframeEncoder(nn.Module):
    """Per-timeframe encoder: feature projection → VSN → LSTM."""

    def __init__(self, n_features: int, seq_len: int, d_model: int,
                 lstm_layers: int, lstm_hidden: int, grn_dropout: float, vsn_dropout: float):
        super().__init__()
        self.seq_len = seq_len

        # Project each feature to d_model, then VSN selects across features
        self.feat_proj = PerVariableProjection(n_features, d_model)
        self.vsn       = VSN(n_features, d_model, dropout=vsn_dropout)
        self.lstm      = nn.LSTM(d_model, lstm_hidden, num_layers=lstm_layers,
                                 batch_first=True, dropout=grn_dropout if lstm_layers > 1 else 0.0)
        self.proj_out  = nn.Linear(lstm_hidden, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_features]
        b, s, f = x.shape

        # Expand each feature to d_model: [batch, seq_len, n_features, d_model]
        x_exp = self.feat_proj(x)                 # [b, s, f, d_model]
        vsn_out = self.vsn(x_exp)                 # [b, s, d_model]

        lstm_out, _ = self.lstm(vsn_out)          # [b, s, lstm_hidden]
        out = self.norm(self.proj_out(lstm_out))  # [b, s, d_model]
        return out


class QuantileNet(nn.Module):
    """Temporal Fusion Transformer for probabilistic price forecasting.

    Supports:
    - Multiple configurable timeframes fused via cross-TF attention
    - Quantile head: price threshold at fixed quantile levels
    - Threshold head: probability of exceeding fixed price thresholds
    - Directions: long / short / both
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        model_cfg   = config['model']
        pred_cfg    = config['prediction']
        tf_cfg      = config['timeframes']
        feat_cfg    = config['features']

        d_model      = model_cfg['d_model']
        n_heads      = model_cfg['n_attention_heads']
        lstm_layers  = model_cfg['lstm_layers']
        lstm_hidden  = model_cfg['lstm_hidden']
        grn_drop     = model_cfg['grn_dropout']
        vsn_drop     = model_cfg['vsn_dropout']
        ff_dim       = model_cfg['ff_dim']
        n_blocks     = model_cfg['n_transformer_blocks']
        n_features   = feat_cfg['num_features']

        self.mode      = pred_cfg['mode']       # quantile | threshold | both
        self.direction = pred_cfg['direction']  # long | short | both

        self.n_horizons   = len(pred_cfg['horizons_hours'])
        self.n_quantiles  = len(pred_cfg.get('quantiles', []))
        self.n_thresholds = len(pred_cfg.get('thresholds_pct', []))
        # Always 2: dir 0 = up excursion (MFE), dir 1 = down excursion (MAE)
        self.n_directions = 2

        # Active timeframes
        TF_KEYS = ['1m', '15m', '1h', '4h', '1d', '1w']
        TF_CFG  = {'1m': 'candles_1m', '15m': 'candles_15m', '1h': 'candles_1h',
                   '4h': 'candles_4h', '1d': 'candles_1d',  '1w': 'candles_1w'}

        self.active_tfs = [tf for tf in TF_KEYS if tf_cfg.get(TF_CFG[tf], 0) > 0]
        self.tf_seq_lens = {tf: tf_cfg[TF_CFG[tf]] for tf in self.active_tfs}

        # Per-TF encoders
        self.tf_encoders = nn.ModuleDict({
            tf: TimeframeEncoder(
                n_features, self.tf_seq_lens[tf], d_model,
                lstm_layers, lstm_hidden, grn_drop, vsn_drop,
            )
            for tf in self.active_tfs
        })

        # Cross-TF attention
        self.pos_enc = PositionalEncoding(d_model, dropout=grn_drop)
        self.blocks  = nn.ModuleList([
            TFBlock(d_model, n_heads, ff_dim, dropout=grn_drop)
            for _ in range(n_blocks)
        ])

        self.post_attn_grn = GRN(d_model, ff_dim, dropout=grn_drop)

        # Output heads (only those needed by mode)
        if self.mode in ('quantile', 'both'):
            assert self.n_quantiles > 0, "quantiles list required for mode=quantile/both"
            self.quantile_head = QuantileHead(d_model, self.n_horizons, self.n_quantiles, self.n_directions)

        if self.mode in ('threshold', 'both'):
            assert self.n_thresholds > 0, "thresholds_pct list required for mode=threshold/both"
            self.threshold_head = ThresholdHead(d_model, self.n_horizons, self.n_thresholds, self.n_directions)

    def forward(self, tf_inputs: dict) -> dict:
        """
        tf_inputs: dict tf_key → Tensor[batch, seq_len, n_features]
        Returns dict with keys 'quantile' and/or 'threshold'.
        """
        encoded = []
        for tf in self.active_tfs:
            x = tf_inputs[tf]                    # [batch, seq_len, n_features]
            encoded.append(self.tf_encoders[tf](x))  # [batch, seq_len, d_model]

        # Concatenate all TF sequences along time axis
        combined = torch.cat(encoded, dim=1)     # [batch, total_seq_len, d_model]
        combined = self.pos_enc(combined)

        for block in self.blocks:
            combined = block(combined)

        # Global average pooling → [batch, d_model]
        pooled = combined.mean(dim=1)
        pooled = self.post_attn_grn(pooled)

        out = {}
        if self.mode in ('quantile', 'both'):
            out['quantile'] = self.quantile_head(pooled)
        if self.mode in ('threshold', 'both'):
            out['threshold'] = self.threshold_head(pooled)

        return out


def upgrade_legacy_state_dict(model: nn.Module, state_dict: dict) -> dict:
    """Convert checkpoints saved with the shared Linear(1, d_model) feat_proj
    to PerVariableProjection.

    The conversion is an exact functional equivalent: every variable's row is
    initialized with the shared direction (old weight [d_model, 1] broadcast to
    [n_vars, d_model], old bias [d_model] likewise). Works for both raw model
    state dicts and EMA/AveragedModel dicts ('module.'-prefixed keys) — the
    match is on the key suffix. Returns the input dict unchanged when no
    legacy feat_proj keys are present.
    """
    upgraded = None
    for name, param in model.state_dict().items():
        if not name.endswith('feat_proj.weight') or name not in state_dict:
            continue
        old_w = state_dict[name]
        if old_w.shape == param.shape:
            continue
        if old_w.ndim == 2 and old_w.shape[1] == 1:
            if upgraded is None:
                upgraded = dict(state_dict)
            n_vars, d_model = param.shape
            upgraded[name] = old_w[:, 0].expand(n_vars, d_model).clone()
            bias_name = name[: -len('weight')] + 'bias'
            old_b = state_dict.get(bias_name)
            if old_b is not None and old_b.ndim == 1:
                upgraded[bias_name] = old_b.expand(n_vars, d_model).clone()
    return upgraded if upgraded is not None else state_dict
