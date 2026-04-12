"""
BaselineComparator — porównuje wytrenowaną sieć z losową bazą.

Oblicza te same metryki co HealthRunner na losowej sieci (random weights)
raz na początku treningu. Służy do oceny czy sieć faktycznie poprawia się
ponad przypadkowy punkt startowy.
"""

import json
import logging
import os
import time

import torch

logger = logging.getLogger('learner')


class BaselineComparator:
    def __init__(self, config: dict, device: str,
                 output_path: str = 'python/diagnostics/baseline.json'):
        self.config = config
        self.device = device
        self.output_path = output_path
        self.baseline: dict | None = None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Załaduj istniejący baseline jeśli jest (np. po restarcie)
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    self.baseline = json.load(f)
                logger.info(f'[Baseline] Loaded existing baseline from {output_path}')
            except Exception:
                pass

    @property
    def computed(self) -> bool:
        return self.baseline is not None

    def compute(self, buffer) -> dict:
        """Oblicza metryki na losowej sieci. Wywołać raz po zapełnieniu buffera."""
        from model.network import TradingDQN

        random_net = TradingDQN(self.config).to(self.device)
        random_net.eval()

        n = min(100, len(buffer))
        sample = buffer.sample(n)
        states = sample[0]

        with torch.no_grad():
            state_list = [states[k] for k in sorted(states.keys())]
            q = random_net(state_list).cpu()

        spread = (q.max(1).values - q.min(1).values)
        greedy = q.argmax(1)
        dominant = greedy.mode().values.item()

        self.baseline = {
            'computed_at': time.time(),
            'q_spread_mean': spread.mean().item(),
            'q_spread_std': spread.std().item(),
            'action_diversity': len(greedy.unique()),
            'dominant_ratio': (greedy == dominant).float().mean().item(),
            'q_sign_positive': (q.max(1).values > 0).float().mean().item(),
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.baseline, f, indent=2)

        logger.info(
            f'[Baseline] Computed: q_spread={self.baseline["q_spread_mean"]:.4f} '
            f'diversity={self.baseline["action_diversity"]}/4 '
            f'dominant_ratio={self.baseline["dominant_ratio"]:.1%}'
        )
        return self.baseline

    def compare(self, trained_metrics: dict) -> dict:
        """Zwraca relative improvement trenowanej sieci vs baseline."""
        if self.baseline is None:
            return {}
        b = self.baseline
        result = {}
        for key in ('q_spread_mean', 'action_diversity', 'dominant_ratio'):
            baseline_val = b.get(key)
            trained_val = trained_metrics.get(key)
            if baseline_val is not None and trained_val is not None and baseline_val != 0:
                result[f'{key}_vs_baseline'] = trained_val / baseline_val
        return result
