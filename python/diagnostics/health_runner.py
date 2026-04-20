"""
HealthRunner — co godzinę robi forward pass na 100 stanach z buffera
i oblicza metryki zdrowia sieci: Q-spread, action diversity, Q-sign consistency.

Wyniki logowane do TensorBoard (prefix Health/) i do health_checks.jsonl.
"""

import json
import logging
import math
import os
import time

import torch
import numpy as np

logger = logging.getLogger('learner')


class HealthRunner:
    def __init__(self, trainer, check_interval_sec: int = 3600,
                 output_path: str = 'python/diagnostics/health_checks.jsonl'):
        self.trainer = trainer
        self.check_interval_sec = check_interval_sec
        self.output_path = output_path
        self._last_check_time: float = 0.0
        self._prev_metrics: dict | None = None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def maybe_run(self) -> dict | None:
        """Wywołać po każdym train_step. Zwraca metryki jeśli wykonał check, None wpp."""
        if time.time() - self._last_check_time < self.check_interval_sec:
            return None
        if len(self.trainer.buffer) < self.trainer.min_buffer_size:
            return None
        self._last_check_time = time.time()
        try:
            result = self._run_check()
            self._log(result)
            return result
        except Exception as e:
            logger.warning(f'[HealthRunner] Check failed: {e}')
            return None

    def _run_check(self) -> dict:
        n = min(100, len(self.trainer.buffer))
        step = self.trainer.step_count

        # Próbkuj stany z buffera
        # Zwraca: (states, actions, rewards, next_states, dones, action_masks,
        #          pos_features, next_pos_features, indices[, is_weights])
        sample = self.trainer.buffer.sample(n)
        states       = sample[0]
        action_masks = sample[5]
        pos_features = sample[6]   # [n, 10] — kontekst pozycji + czas

        network = self.trainer.main_network
        device = self.trainer.device

        network.eval()
        with torch.no_grad():
            state_list = [states[k] for k in sorted(states.keys())]
            q = network(state_list, position_features=pos_features)  # [n, 4]
            value = network._debug_value         # [n, 1]
            advantage = network._debug_advantage # [n, 4]

        network.train()

        q_cpu = q.cpu()
        spread = (q_cpu.max(1).values - q_cpu.min(1).values)

        # Masked Q (tylko dozwolone akcje) — używany też do wyboru dominującej akcji
        q_masked = q_cpu.masked_fill(action_masks.cpu() == 0, float('-inf'))
        spread_masked = (q_masked.max(1).values - q_masked.min(1).values)

        greedy_actions = q_masked.argmax(1)  # respektuje action_masks
        unique_actions = greedy_actions.unique()
        dominant = greedy_actions.mode().values.item()
        dominant_ratio = (greedy_actions == dominant).float().mean().item()
        spread_masked_finite = spread_masked[spread_masked.isfinite()]

        action_names = ['LONG', 'SHORT', 'HOLD', 'CLOSE']

        metrics = {
            'step': step,
            'ts': time.time(),
            'q_spread_mean': spread.mean().item(),
            'q_spread_std': spread.std().item(),
            'q_spread_masked': spread_masked_finite.mean().item() if len(spread_masked_finite) > 0 else 0.0,
            'action_diversity': len(unique_actions),
            'dominant_action': action_names[dominant] if dominant < 4 else str(dominant),
            'dominant_ratio': dominant_ratio,
            'q_sign_positive': (q_cpu.max(1).values > 0).float().mean().item(),
            'value_mean': value.cpu().mean().item() if value is not None else None,
            'advantage_range': (advantage.cpu().max() - advantage.cpu().min()).item() if advantage is not None else None,
            'advantage_std': advantage.cpu().std().item() if advantage is not None else None,
        }

        # Delta vs poprzedni check
        if self._prev_metrics is not None:
            metrics['q_spread_delta'] = metrics['q_spread_mean'] - self._prev_metrics.get('q_spread_mean', 0)
            metrics['diversity_delta'] = metrics['action_diversity'] - self._prev_metrics.get('action_diversity', 0)

        self._prev_metrics = metrics

        # Log do TensorBoard
        writer = self.trainer.writer
        writer.add_scalar('Health/q_spread_mean', metrics['q_spread_mean'], step)
        writer.add_scalar('Health/q_spread_masked', metrics['q_spread_masked'], step)
        writer.add_scalar('Health/action_diversity', metrics['action_diversity'], step)
        writer.add_scalar('Health/dominant_ratio', metrics['dominant_ratio'], step)
        writer.add_scalar('Health/q_sign_positive', metrics['q_sign_positive'], step)
        if metrics['advantage_std'] is not None:
            writer.add_scalar('Health/advantage_std', metrics['advantage_std'], step)
        if metrics['value_mean'] is not None:
            writer.add_scalar('Health/value_mean', metrics['value_mean'], step)

        logger.info(
            f'[HealthRunner] step={step} q_spread={metrics["q_spread_mean"]:.4f} '
            f'diversity={metrics["action_diversity"]}/4 '
            f'dominant={metrics["dominant_action"]}@{metrics["dominant_ratio"]:.1%} '
            f'adv_std={metrics.get("advantage_std", "?")}'
        )
        return metrics

    def _log(self, metrics: dict) -> None:
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
