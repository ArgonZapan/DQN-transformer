"""
TensorBoard Debugger v2 — diagnostic metrics for DQN training.

Metric namespaces (new, separate from the basic ones):
  GradFlow/{layer}         — L2 gradient norm per layer after backward()
  DeadNeurons/{layer}      — % of ReLU activations == 0 in the batch
  UpdateRatio/{layer}      — ||Δw|| / ||w|| after optimizer.step()
  RL/action_entropy_bits   — policy entropy from softmax(Q) [bits]
  RL/q_spread              — mean(max(Q) - min(Q)) per sample
  RL/value_mean            — mean of value stream
  RL/advantage_std         — std of advantage stream
  Dist/q_values            — histogram of Q-values from the last batch
  Dist/rewards             — histogram of rewards from the last batch
  Dist/td_errors           — histogram of TD errors from the last batch
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class TensorBoardDebugger:
    def __init__(self, writer, config: dict):
        self.writer = writer
        tb_cfg = config.get('tensorboard', {})
        self.debug_interval_sec = tb_cfg.get('debug_interval_sec', 30)

        self._hooks: List = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._prev_weights: Dict[str, torch.Tensor] = {}

    # ── Gradient flow ─────────────────────────────────────────────────────
    def log_gradient_flow(self, model: nn.Module, step: int) -> None:
        """Per-layer gradient L2 norm — detects vanishing/exploding gradients."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            norm = param.grad.detach().norm(2).item()
            self.writer.add_scalar(f'GradFlow/{name.replace(".", "/")}', norm, step)

    # ── Dead neurons via forward hooks ────────────────────────────────────
    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to all ReLU layers."""
        self._remove_hooks()
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._activations[name] = out.detach()
        return hook

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def log_dead_neurons(self, step: int) -> None:
        """% of neurons with output == 0 in the batch (dead ReLU)."""
        for name, act in self._activations.items():
            dead_pct = (act == 0).float().mean().item() * 100.0
            self.writer.add_scalar(
                f'DeadNeurons/{name.replace(".", "/")}', dead_pct, step
            )
        self._activations.clear()

    # ── Weight update ratio ───────────────────────────────────────────────
    def snapshot_weights(self, model: nn.Module) -> None:
        """Zapisuje kopię wag PRZED krokiem optymalizatora."""
        self._prev_weights = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def log_weight_updates(self, model: nn.Module, step: int) -> None:
        """||Δw|| / ||w|| per warstwa — stosunek aktualizacji do wagi."""
        for name, param in model.named_parameters():
            if name not in self._prev_weights or not param.requires_grad:
                continue
            delta = (param.detach() - self._prev_weights[name]).norm(2).item()
            w_norm = self._prev_weights[name].norm(2).item() + 1e-8
            self.writer.add_scalar(
                f'UpdateRatio/{name.replace(".", "/")}', delta / w_norm, step
            )

    # ── RL diagnostics ────────────────────────────────────────────────────
    def log_action_entropy(self, q_values: torch.Tensor, step: int) -> None:
        """Entropia polityki z softmax(Q) w bitach. Wyżej = więcej eksploracji."""
        with torch.no_grad():
            probs = torch.softmax(q_values, dim=1)
            entropy = -(probs * torch.log2(probs + 1e-8)).sum(dim=1).mean().item()
        self.writer.add_scalar('RL/action_entropy_bits', entropy, step)

    def log_q_diagnostics(self, q_values: torch.Tensor, step: int) -> None:
        """Q-spread (max-min per próbka) i histogram Q-values."""
        with torch.no_grad():
            spread = (q_values.max(dim=1).values - q_values.min(dim=1).values).mean().item()
        self.writer.add_scalar('RL/q_spread', spread, step)
        self.writer.add_histogram('Dist/q_values', q_values.detach().cpu().numpy(), step)

    def log_value_advantage(
        self,
        value: Optional[torch.Tensor],
        advantage: Optional[torch.Tensor],
        step: int,
    ) -> None:
        """Średnia value stream i std advantage stream."""
        if value is not None:
            self.writer.add_scalar('RL/value_mean', value.detach().mean().item(), step)
        if advantage is not None:
            self.writer.add_scalar('RL/advantage_std', advantage.detach().std().item(), step)

    def log_reward_stats(self, rewards: torch.Tensor, step: int) -> None:
        r = rewards.detach().cpu()
        self.writer.add_scalar('RL/reward_mean', r.mean().item(), step)
        self.writer.add_scalar('RL/reward_std', r.std().item(), step)
        self.writer.add_histogram('Dist/rewards', r.numpy(), step)

    def log_td_histogram(self, td_errors: torch.Tensor, step: int) -> None:
        self.writer.add_histogram('Dist/td_errors', td_errors.detach().cpu().numpy(), step)

    def close(self) -> None:
        self._remove_hooks()
