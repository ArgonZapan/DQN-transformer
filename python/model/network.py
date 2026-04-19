import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Blok konwolucyjny dla pojedynczego timeframe'a.
    Wykrywa lokalne wzorce cenowe, skoki wolumenu, momentum.
    Używa kauzalnego paddingu (tylko w lewo) — brak look-ahead bias.

    Zmiany vs poprzednia wersja:
    - BatchNorm1d zastąpiony LayerNorm: BN w trybie eval używa statystyk z całego
      treningu zamiast bieżącego batcha — to powoduje rozbieżność w Double DQN
      (main.eval() → action selection vs main.train() → backprop).
      LayerNorm normalizuje per-sample, bez trybu train/eval.
    - Brak Global Average Pooling: GAP redukował sekwencję N świec do 1 tokenu,
      co czyniło Transformer zbędnym (self-attention nad 4 tokenami = ważona suma).
      Teraz zwracamy pełną sekwencję [batch, seq_len, conv_filters] — Transformer
      dostaje wszystkie świece ze wszystkich TF jako tokeny.
    """

    def __init__(self, num_features, conv_filters, kernel_size, dropout):
        super().__init__()
        # padding=0 — pad ręcznie tylko po lewej stronie (przeszłość)
        self.conv = nn.Conv1d(num_features, conv_filters, kernel_size, padding=0)
        self.causal_pad = kernel_size - 1  # ile zer dodać po lewej
        self.ln = nn.LayerNorm(conv_filters)  # per-sample, niezależny od trybu train/eval
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)                  # [batch, features, seq_len]
        x = F.pad(x, (self.causal_pad, 0))     # kauzalny padding
        x = self.conv(x)                        # [batch, conv_filters, seq_len]
        x = x.transpose(1, 2)                  # [batch, seq_len, conv_filters]
        x = self.ln(x)                          # LayerNorm po osi conv_filters
        x = F.gelu(x)
        x = self.dropout(x)
        # Zwracamy pełną sekwencję [batch, seq_len, conv_filters] — bez GAP
        return x


class TransformerEncoderBlock(nn.Module):
    """Blok Transformer Encoder z Multi-Head Attention i residual connections."""

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
    Główna sieć Conv1D + Transformer + Dueling DQN.

    Architektura (po poprawkach issue #31):
    1. N Conv1D bloków (jeden per timeframe) → [batch, seq_tf, conv_filters]
       - LayerNorm zamiast BatchNorm (brak rozbieżności train/eval w Double DQN)
       - Brak GAP — sekwencja zachowana w całości
    2. Konkatenacja sekwencji ze wszystkich TF → [batch, total_seq, conv_filters]
       - Transformer dostaje ~130 tokenów (świece) zamiast 4 (jeden per TF)
       - Usunięta zbędna projekcja identity Linear(d→d)
    3. M Transformer Encoder bloków → [batch, total_seq, conv_filters]
    4. Global Average Pooling po osi sekwencji → [batch, conv_filters]
    5. Trunk Dense + concat z cechami pozycji → Dueling heads
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

        # Normalizacja wejścia — wyrównuje skale 8 cech zanim trafią do Conv1D.
        # MACD jest unbounded, volume spikes >100x, ceny z-score ±3 — bez normy
        # cechy o dużej skali dominują gradienty.
        # LayerNorm zamiast BatchNorm: normalizuje per-sample, brak trybu train/eval.
        self.input_norm = nn.LayerNorm(self.num_features)

        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(self.num_features, self.conv_filters, self.kernel_size, self.dropout)
            for _ in range(self.num_timeframes)
        ])

        concat_dim = self.conv_filters * self.num_timeframes

        if self.n_transformer_blocks > 0:
            # Projekcja identity Linear(d→d) usunięta — nie zwiększa ekspresywności
            # bez nieliniowości (issue #31, problem 3)
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(self.conv_filters, self.n_heads, self.ff_dim, self.dropout)
                for _ in range(self.n_transformer_blocks)
            ])
            trunk_dim = self.conv_filters
        else:
            self.transformer_blocks = nn.ModuleList()
            trunk_dim = concat_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # Gałąź pozycji: [is_long, is_short, unrealized_pnl, bars_in_trade] → 32
        self.pos_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
        )

        # Dueling Architecture: trunk(256) + pos(32) = 288
        self.value_stream = nn.Linear(288, 1)
        self.advantage_stream = nn.Linear(288, self.num_actions)

        # Inicjalizacja głowic do bliskich zera — zapobiega wczesnemu action collapse
        nn.init.uniform_(self.advantage_stream.weight, -0.01, 0.01)
        nn.init.zeros_(self.advantage_stream.bias)
        nn.init.uniform_(self.value_stream.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_stream.bias)

    def forward(self, states, action_mask=None, position_features=None):
        """
        states: dict z kluczami timeframe'ów, każdy [batch, seq_len, num_features]
                lub lista tensorów w kolejności timeframe_keys
        action_mask: [batch, num_actions] binary mask (1=dozwolone, 0=zablokowane)
        position_features: [batch, 4] — [is_long, is_short, unrealized_pnl, bars_in_trade]
                           None → zerowy wektor (brak kontekstu pozycji)
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
            # Połącz sekwencje ze wszystkich TF wzdłuż osi czasu
            # → [batch, total_seq, conv_filters]  (np. 30+24+48+30 = 132 tokenów)
            x = torch.cat(conv_outputs, dim=1)
            for block in self.transformer_blocks:
                x = block(x)
            # GAP po osi sekwencji — po Transformerze, nie przed
            x = x.mean(dim=1)                         # [batch, conv_filters]
        else:
            # Bez Transformera: GAP per TF, potem concat
            x = torch.cat([c.mean(dim=1) for c in conv_outputs], dim=1)

        x = self.trunk(x)  # [batch, 256]

        # Konkatencja z cechami pozycji
        if position_features is not None:
            pos = self.pos_fc(position_features)   # [batch, 32]
        else:
            pos = torch.zeros(x.shape[0], 32, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pos], dim=1)            # [batch, 288]

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q = V + (A - mean(A)) — odejmowanie mean(A) identyfikuje model jednoznacznie
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # Przechowaj dla debuggera (bez narzutu gdy debugger nieaktywny)
        self._debug_value = value
        self._debug_advantage = advantage

        if action_mask is not None:
            # Zablokuj niedozwolone akcje ustawiając Q na -inf
            # zamiast 0, bo 0 mogłoby być wyższe niż Q dozwolonej akcji
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

        return q_values

    def reset_noise(self):
        """Reset szumu w NoisyLinear (jeśli używane)."""
        for module in self.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
