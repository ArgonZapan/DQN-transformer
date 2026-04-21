"""
AttentionMonitor — forward hooks na blokach transformera.

Mierzy per-head:
  - entropię attention (0 = skupiona na 1 tokenie, log(n_tokens) = uniform/losowa)
  - dywergencję KL między głowami (niska = głowy identyczne = redundancja)
  - max weight per row (koncentracja uwagi)

Wymaga network.py z average_attn_weights=False w MultiheadAttention.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger('learner')


class AttentionMonitor:
    def __init__(self, network: nn.Module, writer):
        self.network = network
        self.writer = writer
        self._hooks: list = []
        self._attn_cache: dict[int, torch.Tensor] = {}   # block_idx → [B, heads, seq, seq]

    def register(self) -> None:
        """Podpina hooki na wszystkie bloki TransformerEncoderBlock."""
        self._remove_hooks()
        for name, module in self.network.named_modules():
            if type(module).__name__ == 'TransformerEncoderBlock':
                # Wyciągnij numer bloku z nazwy (np. transformer_blocks.0)
                parts = name.split('.')
                try:
                    idx = int(parts[-1])
                except (ValueError, IndexError):
                    idx = len(self._hooks)
                self._hooks.append(
                    module.attn.register_forward_hook(self._make_hook(idx))
                )
        logger.info(f'[AttentionMonitor] Registered {len(self._hooks)} hooks')

    def _make_hook(self, block_idx: int):
        def hook(module, inp, output):
            # output = (attn_output, attn_weights)
            # attn_weights kształt zależy od average_attn_weights:
            #   True (domyślne):  [B, seq, seq]       — uśrednione po głowach
            #   False:            [B, heads, seq, seq] — per-head
            if isinstance(output, tuple) and len(output) == 2:
                weights = output[1]
                if weights is not None:
                    self._attn_cache[block_idx] = weights.detach().cpu()
        return hook

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def log_diagnostics(self, step: int) -> dict:
        """Oblicza i loguje metryki attention. Wywołać po forward passie."""
        if not self._attn_cache:
            return {}

        result = {}
        for block_idx, weights in self._attn_cache.items():
            # weights: [B, seq, seq] (uśrednione) lub [B, heads, seq, seq]
            if weights.dim() == 3:
                # Uśrednione po głowach — traktuj jako 1 głowę
                weights = weights.unsqueeze(1)   # [B, 1, seq, seq]

            # weights: [B, heads, seq, seq]
            B, H, S, _ = weights.shape
            # Uśrednij po batch
            w = weights.mean(0)   # [heads, seq, seq]

            # Entropia per głowa — wektoryzowane: jeden sync zamiast H
            p = w.clamp(min=1e-9)
            p = p / p.sum(dim=-1, keepdim=True)
            ent_per_head = -(p * p.log()).sum(dim=-1).mean(dim=-1)   # [heads]
            entropies = ent_per_head.tolist()
            for h, ent in enumerate(entropies):
                self.writer.add_scalar(f'Attention/block{block_idx}/head{h}_entropy', ent, step)
            mean_ent = sum(entropies) / len(entropies)
            self.writer.add_scalar(f'Attention/block{block_idx}/mean_entropy', mean_ent, step)

            # Symetryczna KL między wszystkimi parami głów — jeden sync na blok
            log_p = p.log()
            # KL(p_i || p_j) per row, mean po queries:  sum_k p_i(k) * (log p_i(k) - log p_j(k))
            # → tensor [H, H]
            kl_pq = (p.unsqueeze(1) * (log_p.unsqueeze(1) - log_p.unsqueeze(0))).sum(dim=-1).mean(dim=-1)
            sym_kl = (kl_pq + kl_pq.t()) / 2.0
            sym_kl.fill_diagonal_(0.0)
            max_kl = sym_kl.max().item()
            self.writer.add_scalar(f'Attention/block{block_idx}/head_divergence', max_kl, step)

            # Koncentracja: max weight per row, mean po batch i queries
            max_w = weights.max(dim=-1).values.mean().item()
            self.writer.add_scalar(f'Attention/block{block_idx}/max_weight', max_w, step)

            result[f'block{block_idx}_mean_entropy'] = mean_ent
            result[f'block{block_idx}_head_divergence'] = max_kl

        self._attn_cache.clear()
        return result

    def remove(self) -> None:
        self._remove_hooks()
