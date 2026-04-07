import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Blok konwolucyjny dla pojedynczego timeframe'a.
    Wykrywa lokalne wzorce cenowe, skoki wolumenu, momentum."""

    def __init__(self, num_features, conv_filters, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(num_features, conv_filters, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(conv_filters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features] -> transpose to [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Global Average Pooling: [batch, filters, seq_len] -> [batch, filters]
        x = x.mean(dim=2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Blok Transformer Encoder z Multi-Head Attention i residual connections."""

    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
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

    Architektura:
    1. N Conv1D bloków (jeden per timeframe) -> wyciąganie lokalnych wzorców
    2. Konkatenacja + projekcja -> reshape do sekwencji tokenów
    3. M Transformer Encoder bloków -> zależności między timeframe'ami
    4. Global Average Pooling -> Dense
    5. Dueling: Value stream + Advantage stream -> Q(s,a) = V(s) + (A(s,a) - mean(A))
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

        self.timeframe_keys = sorted(timeframes_cfg.keys())
        self.num_timeframes = len(self.timeframe_keys)

        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(self.num_features, self.conv_filters, self.kernel_size, self.dropout)
            for _ in range(self.num_timeframes)
        ])

        concat_dim = self.conv_filters * self.num_timeframes

        if self.n_transformer_blocks > 0:
            self.projection = nn.Linear(self.conv_filters, self.conv_filters)
            self.transformer_blocks = nn.ModuleList([
                TransformerEncoderBlock(self.conv_filters, self.n_heads, self.ff_dim, self.dropout)
                for _ in range(self.n_transformer_blocks)
            ])
            trunk_dim = self.conv_filters
        else:
            self.projection = None
            self.transformer_blocks = nn.ModuleList()
            trunk_dim = concat_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Dueling Architecture: oddzielenie wartości stanu od przewagi akcji
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, self.num_actions)

    def forward(self, states, action_mask=None):
        """
        states: dict z kluczami timeframe'ów, każdy [batch, seq_len, num_features]
                lub lista tensorów w kolejności timeframe_keys
        action_mask: [batch, num_actions] binary mask (1=dozwolone, 0=zablokowane)
        """
        if isinstance(states, dict):
            tf_tensors = [states[k] for k in self.timeframe_keys]
        else:
            tf_tensors = states

        conv_outputs = []
        for i, tf_input in enumerate(tf_tensors):
            conv_out = self.conv_blocks[i](tf_input)
            conv_outputs.append(conv_out)

        if self.n_transformer_blocks > 0:
            tokens = torch.stack([self.projection(c) for c in conv_outputs], dim=1)
            x = tokens
            for block in self.transformer_blocks:
                x = block(x)
            x = x.mean(dim=1)
        else:
            x = torch.cat(conv_outputs, dim=1)

        x = self.trunk(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q = V + (A - mean(A)) — odejmowanie mean(A) identyfikuje model jednoznacznie
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if action_mask is not None:
            # Zablokuj niedozwolone akcje ustawiając Q na -inf
            # zamiast 0, bo 0 mogłoby być wyższe niż Q dozwolonej akcji
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))

        return q_values

    def get_confidence(self, q_values):
        """Pewność modelu na podstawie rozrzutu Q-values."""
        sorted_q = torch.sort(q_values, dim=1, descending=True)
        confidence = sorted_q.values[:, 0] - sorted_q.values[:, 1]
        return torch.sigmoid(confidence)

    def reset_noise(self):
        """Reset szumu w NoisyLinear (jeśli używane)."""
        for module in self.modules():
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
