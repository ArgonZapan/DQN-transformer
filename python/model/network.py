import torch
import torch.nn as nn
import torch.nn.functional as F

from quantilenet import POSITION_BASE_DIM, extended_position_dim


class Conv1DBlock(nn.Module):
    """Convolutional block for a single timeframe.
    Detects local price patterns, volume spikes, momentum.
    Uses causal padding (left-only) — no look-ahead bias.

    Changes vs previous version:
    - BatchNorm1d replaced by LayerNorm: BN in eval mode uses training-wide statistics
      instead of the current batch — causes divergence in Double DQN
      (main.eval() for action selection vs main.train() for backprop).
      LayerNorm normalizes per-sample, independent of train/eval mode.
    - No Global Average Pooling: GAP reduced N-candle sequence to 1 token,
      making the Transformer redundant (self-attention over 4 tokens = weighted sum).
      Now returns the full sequence [batch, seq_len, conv_filters] — Transformer
      receives all candles from all timeframes as tokens.
    """

    def __init__(self, num_features, conv_filters, kernel_size, dropout):
        super().__init__()
        # padding=0 — pad manually on the left side only (past candles)
        self.conv = nn.Conv1d(num_features, conv_filters, kernel_size, padding=0)
        self.causal_pad = kernel_size - 1  # number of zeros to prepend
        self.ln = nn.LayerNorm(conv_filters)  # per-sample, train/eval agnostic
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)                  # [batch, features, seq_len]
        x = F.pad(x, (self.causal_pad, 0))     # causal padding
        x = self.conv(x)                        # [batch, conv_filters, seq_len]
        x = x.transpose(1, 2)                  # [batch, seq_len, conv_filters]
        x = self.ln(x)                          # LayerNorm over conv_filters axis
        x = F.gelu(x)
        x = self.dropout(x)
        # Return full sequence [batch, seq_len, conv_filters] — no GAP here
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder block with Multi-Head Attention and residual connections."""

    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class TradingDQN(nn.Module):
    """
    Main network: Conv1D + Transformer + Dueling DQN.

    Architecture (after fixes from issue #31):
    1. N Conv1D blocks (one per timeframe) → [batch, seq_tf, conv_filters]
       - LayerNorm instead of BatchNorm (no train/eval divergence in Double DQN)
       - No GAP — full sequence preserved
    2. Concatenate sequences from all timeframes → [batch, total_seq, conv_filters]
       - Transformer receives ~130 tokens (candles) instead of 4 (one per TF)
       - Removed redundant identity projection Linear(d→d)
    3. M Transformer Encoder blocks → [batch, total_seq, conv_filters]
    4. Global Average Pooling over sequence axis → [batch, conv_filters]
    5. Trunk Dense + concat with position features → Dueling heads
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config['model']
        features_cfg = config['features']
        training_cfg = config['training']
        timeframes_cfg = config['timeframes']

        self.num_features = features_cfg['num_features']
        self.num_actions = model_cfg['num_actions']
        self.conv_filters = model_cfg['conv1d_filters']
        self.kernel_size = model_cfg['conv_kernel_size']
        self.n_transformer_blocks = model_cfg['n_transformer_blocks']
        self.n_heads = model_cfg['n_attention_heads']
        self.ff_dim = model_cfg['ff_dim']
        self.dropout = training_cfg['dropout']

        self.timeframe_keys = [k for k in sorted(timeframes_cfg.keys()) if timeframes_cfg[k] > 0]
        self.num_timeframes = len(self.timeframe_keys)

        # Input normalization — equalizes scales of all features before Conv1D.
        # MACD is unbounded, volume spikes >100x, price z-score ±3 — without normalization
        # high-scale features dominate gradients.
        # LayerNorm instead of BatchNorm: normalizes per-sample, no train/eval dependency.
        self.input_norm = nn.LayerNorm(self.num_features)

        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(self.num_features, self.conv_filters, self.kernel_size, self.dropout)
            for _ in range(self.num_timeframes)
        ])

        concat_dim = self.conv_filters * self.num_timeframes

        if self.n_transformer_blocks > 0:
            # Identity projection Linear(d→d) removed — adds no expressiveness
            # without nonlinearity (issue #31, problem 3)
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(self.conv_filters, self.n_heads, self.ff_dim, self.dropout)
                for _ in range(self.n_transformer_blocks)
            ])
            trunk_dim = self.conv_filters
        else:
            self.transformer_blocks = nn.ModuleList()
            trunk_dim = concat_dim

        trunk_h1 = self.conv_filters * 4  # e.g. 64→256
        trunk_h2 = self.conv_filters * 2  # e.g. 64→128
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, trunk_h1),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(trunk_h1, trunk_h2),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # Position branch: 10 base features (position state + time encodings) plus
        # optional prerolled QuantileNet features (e.g. 30 = 3 horizons × 5 quantiles × 2 dirs).
        # When [quantilenet].enabled is false the extra slots are zeros, so the
        # input shape stays constant and checkpoints remain valid across toggles.
        self.position_dim = extended_position_dim(config)
        self.pos_fc = nn.Sequential(
            nn.Linear(self.position_dim, 32),
            nn.GELU(),
        )

        # Dueling Architecture: trunk(trunk_h2) + pos(32)
        dueling_dim = trunk_h2 + 32
        self.value_stream = nn.Linear(dueling_dim, 1)
        self.advantage_stream = nn.Linear(dueling_dim, self.num_actions)

        # Initialize heads close to zero — prevents early action collapse
        nn.init.uniform_(self.advantage_stream.weight, -0.01, 0.01)
        nn.init.zeros_(self.advantage_stream.bias)
        nn.init.uniform_(self.value_stream.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_stream.bias)

    def forward(self, states, action_mask=None, position_features=None):
        """
        states: dict with timeframe keys, each [batch, seq_len, num_features]
                or list of tensors in timeframe_keys order
        action_mask: [batch, num_actions] binary mask (1=allowed, 0=blocked)
        position_features: [batch, 10] — [is_long, is_short, unrealized_pnl, hold_6h, hold_48h,
                                          sin_hour, cos_hour, sin_week, cos_week, is_weekend]
                           None → zero vector (no position context)
        """
        if isinstance(states, dict):
            tf_tensors = [states[k] for k in self.timeframe_keys]
        else:
            tf_tensors = states

        conv_outputs = []
        for i, tf_input in enumerate(tf_tensors):
            tf_input = self.input_norm(tf_input)       # [batch, seq_len, num_features]
            conv_out = self.conv_blocks[i](tf_input)   # [batch, seq_tf, conv_filters]
            conv_outputs.append(conv_out)

        if self.n_transformer_blocks > 0:
            # Concatenate sequences from all timeframes along the time axis
            # → [batch, total_seq, conv_filters]  (e.g. 60+32+48+14 = 154 tokens)
            x = torch.cat(conv_outputs, dim=1)
            for block in self.transformer_blocks:
                x = block(x)
            # GAP over sequence axis — after Transformer, not before
            x = x.mean(dim=1)                         # [batch, conv_filters]
        else:
            # No Transformer: GAP per TF, then concat
            x = torch.cat([c.mean(dim=1) for c in conv_outputs], dim=1)

        x = self.trunk(x)

        # Concatenate with position features
        if position_features is not None:
            pos = self.pos_fc(position_features)
        else:
            pos = torch.zeros(x.shape[0], 32, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pos], dim=1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q = V + (A - mean(A)) — subtracting mean(A) uniquely identifies the model
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # Store for debugger (zero overhead when debugger is inactive)
        self._debug_value = value
        self._debug_advantage = advantage

        if action_mask is not None:
            # Block illegal actions by setting Q to -inf
            # (not 0, because 0 could be higher than a legal action's Q-value)
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

        return q_values

    def reset_noise(self):
        """Reset noise in NoisyLinear layers (if used)."""
        for module in self.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
