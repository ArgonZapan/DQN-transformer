import os
import logging
import glob
import queue
from datetime import datetime
from collections import defaultdict, deque
import time as time_module

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model.network import TradingDQN
from training.replay_buffer import ReplayBuffer
from training.prioritized_buffer import PrioritizedReplayBuffer, DualPrioritizedBuffer
from training.debugger import TensorBoardDebugger
from diagnostics.metric_logger import MetricLogger
from diagnostics.alert_system import AlertSystem
from diagnostics.attention_monitor import AttentionMonitor
from diagnostics.training_report import TrainingReport
from quantilenet import QuantileFeatureLoader, extended_position_dim

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['learner']['device']

        training_cfg = config['training']
        self.gamma = training_cfg['gamma']
        self.return_mode = training_cfg.get('return_mode', 'mc').lower()
        n_step = training_cfg.get('n_step', 1)
        # gamma_n — bootstrap multiplier in Bellman targets
        # mc:    done=True everywhere → (1-done)*gamma_n = 0 → gamma_n irrelevant
        # nstep: targets = R_t^n + (1-done) * gamma^n * Q(s_{t+n})
        # td:    targets = r_t   + (1-done) * gamma   * Q(s_{t+1})
        if self.return_mode == 'nstep':
            self.gamma_n = self.gamma ** n_step
        else:
            self.gamma_n = self.gamma
        logger.info(f"[Trainer] return_mode={self.return_mode}, n_step={n_step}, gamma_n={self.gamma_n:.6f}")
        self.lr = training_cfg['lr']
        self.batch_size = training_cfg['batch_size']
        self.min_buffer_size = training_cfg['min_buffer_size']
        self.target_update_interval = training_cfg['target_update_interval']
        self.epsilon_start = training_cfg['epsilon_start']
        self.epsilon_end = training_cfg['epsilon_end']
        self.epsilon_decay_fraction = training_cfg['epsilon_decay_fraction']
        self.checkpoint_interval = training_cfg['checkpoint_interval']
        self.keep_last_n = training_cfg['keep_last_n_checkpoints']
        self.seed = training_cfg['seed']

        if self.seed >= 0:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.main_network = TradingDQN(config).to(self.device)
        self.target_network = TradingDQN(config).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        # Mixed Precision (FP16) — Ampere GPU (3090) has tensor cores; ~2x speedup
        # CPU does not support GradScaler — disabled automatically
        self.use_amp = self.device != 'cpu' and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("[Trainer] Mixed precision (AMP) enabled")

        # torch.compile — moved to end of __init__ (after checkpoint load),
        # because OptimizedModule adds `_orig_mod.` prefix to state_dict, breaking load.
        self._compile_enabled = config.get('training', {}).get('compile_model', False)

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='none')

        # CosineAnnealing LR scheduler synchronized with epsilon decay.
        # T_max = number of training steps in the decay phase (epsilon_decay_fraction * buffer_capacity).
        # Active only when lr_scheduler = "cosine" in config.
        lr_sched_cfg = training_cfg.get('lr_scheduler', 'none').lower()
        if lr_sched_cfg == 'cosine':
            total_decay_steps = int(training_cfg['epsilon_decay_fraction'] * training_cfg['buffer_capacity'])
            lr_min = training_cfg.get('lr_min', self.lr * 0.1)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_decay_steps), eta_min=lr_min
            )
            logger.info(f"[Trainer] LR scheduler: CosineAnnealing T_max={total_decay_steps}, lr {self.lr} → {lr_min}")
        else:
            self.lr_scheduler = None

        # Pre-allocated constant masks for Double DQN (previously created every step)
        self._mask_in_pos = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=self.device)
        self._mask_no_pos = torch.tensor([1, 1, 1, 0], dtype=torch.float32, device=self.device)

        # Experience queue: ZMQ thread only enqueues (fast), training thread
        # drains at the start of each train_step — eliminates GIL contention
        self._experience_queue: queue.Queue = queue.Queue(maxsize=50000)

        # QuantileNet feature loader — must be built before the buffer because
        # the buffer's pos_features dim depends on the configured quantile dim
        # (extended_position_dim reads the same config keys).
        self.quantile_loader = QuantileFeatureLoader(config)
        self.position_dim = extended_position_dim(config)

        per_cfg = config.get('per', {})
        self.use_per = per_cfg.get('alpha', 0) > 0
        self.buffer = self._build_buffer()
        if isinstance(self.buffer, DualPrioritizedBuffer):
            logger.info(f"[Trainer] Using DualPrioritizedBuffer (positive_ratio={per_cfg['positive_ratio']})")

        self.step_count = 0
        self.epsilon = self.epsilon_start
        self.last_loss = 0.0
        self._paused = False
        self._epsilon_reset_end_step: int | None = None
        self._epsilon_reset_target:   float | None = None
        self._logged_training_start = False
        self._sps_step_anchor = 0
        self._sps_time_anchor = time_module.time()
        self._timing_logged = 0  # number of times timing diagnostics have been logged
        
        # TensorBoard setup
        tensorboard_cfg = config.get('tensorboard', {})
        tensorboard_dir = tensorboard_cfg.get('log_dir', 'runs')
        self.log_interval_sec = tensorboard_cfg.get('log_interval_sec', 10)
        self.histogram_interval_sec = tensorboard_cfg.get('histogram_interval_sec', 60)
        self.enable_actor_metrics = tensorboard_cfg.get('enable_actor_metrics', True)
        
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(tensorboard_dir, f"train_{run_name}"))
        logger.info(f"[Trainer] TensorBoard logging to {tensorboard_dir}/train_{run_name} (interval={self.log_interval_sec}s)")
        
        # Timestamp of last TensorBoard log
        self._last_tb_log_time = time_module.time()
        self._last_histogram_log_time = time_module.time()
        self._last_hourly_ckpt_time = time_module.time()
        
        # Debugger v2
        self.debugger = TensorBoardDebugger(self.writer, config)
        self._last_debug_time = 0.0

        # Diagnostics module
        diag_dir = 'python/diagnostics'
        self.metric_logger = MetricLogger(os.path.join(diag_dir, 'metrics.jsonl'), flush_every=50)
        self.alert_system = AlertSystem(config)
        self.training_report = TrainingReport(config)
        # Attention monitor — hooks call .cpu() on EVERY forward pass (GPU sync).
        # Registered dynamically only within the debug_now window, then removed.
        self.attention_monitor = AttentionMonitor(self.main_network, self.writer)
        self._last_advantage_std: float | None = None
        self._last_health: dict | None = None

        # Metric accumulators — GPU tensors, one sync per log_interval instead of 8+/step
        self._num_actions = config['model']['num_actions']
        self._gpu_acc = self._make_gpu_accumulator()
        self._gpu_acc_steps = 0
        
        # Per-symbol actor metrics
        self._actor_metrics = defaultdict(lambda: {
            'transactions': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'episode_pnl_sum': 0.0,
            'episode_count': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_losses': 0,
            # Episode statistics (from actor via ZMQ)
            'episode_length_sum': 0,
            'episode_length_count': 0,
            'episode_trade_wins': 0,
            'episode_trade_losses': 0,
            # Extended trade statistics
            'profit_sum': 0.0,          # sum of profitable trades
            'loss_sum': 0.0,            # sum of losing trades (abs)
            'win_hold_steps': 0,        # total hold steps for winning positions
            'loss_hold_steps': 0,       # total hold steps for losing positions
            'mfe_sum': 0.0,             # sum of MFE (Maximum Favorable Excursion)
            'long_opens': 0,
            'short_opens': 0,
            'action_long': 0,
            'action_short': 0,
            'action_hold': 0,
            'action_close': 0,
            # Last 200 episode PnL deque — for Sharpe/Sortino/MaxDD
            'episode_pnl_deque': deque(maxlen=200),
            # Last 5000 gap_steps deque — for median/p10/p90 time between trades
            'gap_steps_deque': deque(maxlen=5000),
        })
        
        # EMA for Q-values
        self._q_values_ema = None
        self._q_ema_decay = 0.99
        
        # State reset in reset_state(); initialized here to ensure fields
        # exist before train_step has a chance to set them.
        self._logged_halfway = False
        self._last_q = None
        self._last_rewards = None
        self._last_td = None
        self._last_reward_mean = None

        self._load_checkpoint_if_exists()

        # torch.compile AFTER checkpoint load — OptimizedModule adds `_orig_mod.`
        # prefix to state_dict, so compiling before load_state_dict breaks loading.
        if self._compile_enabled and hasattr(torch, 'compile') and self.device != 'cpu':
            # Windows has no Triton → inductor backend fails. aot_eager works everywhere
            # (smaller gain ~5-10% instead of 20-40%, but no Triton requirement).
            compile_backend = config.get('training', {}).get('compile_backend', 'aot_eager')
            try:
                self.main_network = torch.compile(self.main_network, backend=compile_backend, dynamic=False)
                self.target_network = torch.compile(self.target_network, backend=compile_backend, dynamic=False)
                logger.info(f"[Trainer] torch.compile enabled (backend={compile_backend})")
            except Exception as e:
                logger.warning(f"[Trainer] torch.compile failed, falling back to eager: {e}")

        # Set SPS anchor after checkpoint load (step_count may be != 0 on resume)
        self._sps_step_anchor = self.step_count
        self._sps_time_anchor = time_module.time()
        logger.info(f"[Trainer] Init: batch={self.batch_size}, min_buf={self.min_buffer_size}, steps={self.step_count}, resumed={self.step_count > 0}")

    def _load_checkpoint_if_exists(self):
        resume = self.config['training'].get('resume_from_checkpoint', '')
        if resume and os.path.exists(resume):
            self._load_checkpoint(resume)
            return

        ckpt_dir = os.path.join('python', 'checkpoints')

        # Collect ALL checkpoints and select the one with the highest step as model
        candidates = []
        for name in ('shutdown_checkpoint.pt', 'hourly_checkpoint.pt'):
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                try:
                    c = torch.load(p, map_location='cpu', weights_only=False)
                    candidates.append((c.get('step', 0), p))
                except Exception as e:
                    logger.warning(f"Could not read {p}: {e}")

        for p in sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_step_*.pt'))):
            try:
                c = torch.load(p, map_location='cpu', weights_only=False)
                candidates.append((c.get('step', 0), p))
            except Exception as e:
                logger.warning(f"Could not read {p}: {e}")

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_model_path = candidates[0][1]
        logger.info(f"[Checkpoint] Loading model from: {best_model_path} (step {candidates[0][0]})")
        self._load_checkpoint(best_model_path, load_buffer=False)

        # Load buffer from the most recent file that contains it (shutdown > hourly)
        buffer_candidates = []
        for name in ('shutdown_checkpoint.pt', 'hourly_checkpoint.pt'):
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                try:
                    c = torch.load(p, map_location='cpu', weights_only=False)
                    if 'buffer' in c:
                        buffer_candidates.append((c.get('step', 0), p))
                except Exception:
                    pass
        if buffer_candidates:
            buffer_candidates.sort(key=lambda x: x[0], reverse=True)
            buf_path = buffer_candidates[0][1]
            logger.info(f"[Checkpoint] Loading replay buffer from: {buf_path} (step {buffer_candidates[0][0]})")
            self._load_buffer(buf_path)

    def _upgrade_pos_fc_for_quantiles(self, state_dict):
        target = self.main_network.state_dict().get('pos_fc.0.weight')
        ckpt_w = state_dict.get('pos_fc.0.weight')
        if target is None or ckpt_w is None:
            return None
        if target.shape[0] != ckpt_w.shape[0] or target.shape[1] <= ckpt_w.shape[1]:
            return None
        new_w = torch.zeros_like(target)
        new_w[:, :ckpt_w.shape[1]] = ckpt_w
        upgraded = dict(state_dict)
        upgraded['pos_fc.0.weight'] = new_w
        return upgraded

    def _load_checkpoint(self, path, load_buffer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.main_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # Common case: pre-quantilenet checkpoint has pos_fc.0.weight
            # of shape [32, 10]; current network expects [32, 10+quantile_dim].
            # Pad the input dim with zeros so legacy base weights are kept and
            # quantile contribution starts at zero, then retry.
            patched = self._upgrade_pos_fc_for_quantiles(checkpoint['model_state_dict'])
            if patched is not None:
                try:
                    self.main_network.load_state_dict(patched)
                    self.target_network.load_state_dict(patched)
                    # Optimizer Adam state (m, v) was tied to old pos_fc shape;
                    # discard it so the new param starts with fresh momentum.
                    checkpoint.pop('optimizer_state_dict', None)
                    logger.info("[Checkpoint] Upgraded legacy pos_fc weights for quantile features (zero-padded). Optimizer state reset.")
                except RuntimeError as e2:
                    logger.warning(f"[Checkpoint] Architecture mismatch — starting with fresh weights. ({e2})")
                    return
            else:
                logger.warning(f"[Checkpoint] Architecture mismatch — starting with fresh weights. ({e})")
                return
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.last_loss = checkpoint.get('loss', 0.0)
        self._epsilon_reset_end_step = checkpoint.get('epsilon_reset_end_step', None)
        self._epsilon_reset_target   = checkpoint.get('epsilon_reset_target',   None)
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.lr
        if load_buffer and 'buffer' in checkpoint:
            buf_state = checkpoint['buffer']
            buf_size = buf_state.get('size', buf_state.get('main', {}).get('size', '?'))
            logger.info(f"[Checkpoint] Restoring replay buffer ({buf_size} experiences)...")
            self.buffer.load_state(buf_state)
            logger.info(f"[Checkpoint] Buffer restored: {len(self.buffer)} experiences")
        logger.info(f"Checkpoint loaded from {path} (step {self.step_count})")

    def _load_buffer(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'buffer' in checkpoint:
            buf_state = checkpoint['buffer']
            buf_size = buf_state.get('size', buf_state.get('main', {}).get('size', '?'))
            logger.info(f"[Checkpoint] Restoring replay buffer ({buf_size} experiences)...")
            self.buffer.load_state(buf_state)
            logger.info(f"[Checkpoint] Buffer restored: {len(self.buffer)} experiences")

    def save_checkpoint(self, path=None, include_buffer=False):
        if path is None:
            os.makedirs(os.path.join('python', 'checkpoints'), exist_ok=True)
            path = os.path.join('python', 'checkpoints', f'checkpoint_step_{self.step_count}.pt')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Always save the eager model state — torch.compile adds `_orig_mod.`
        # prefix to state_dict, which breaks loading into an eager model (compile happens after load).
        eager_model = getattr(self.main_network, '_orig_mod', self.main_network)
        payload = {
            'step': self.step_count,
            'model_state_dict': eager_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.last_loss,
            'epsilon': self.epsilon,
        }
        if self.lr_scheduler is not None:
            payload['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        if self._epsilon_reset_end_step is not None:
            payload['epsilon_reset_end_step'] = self._epsilon_reset_end_step
            payload['epsilon_reset_target']   = self._epsilon_reset_target
        if include_buffer:
            buf_size = len(self.buffer)
            logger.info(f"[Checkpoint] Serializing buffer ({buf_size} experiences)...")
            t0 = time_module.time()
            payload['buffer'] = self.buffer.get_state()
            elapsed = time_module.time() - t0
            logger.info(f"[Checkpoint] Buffer serialized in {elapsed:.1f}s")

        torch.save(payload, path)
        logger.info(f"Checkpoint saved to {path} (buffer={'yes' if include_buffer else 'no'})")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        checkpoint_dir = os.path.join('python', 'checkpoints')
        pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.pt')
        checkpoints = sorted(glob.glob(pattern))
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            os.remove(old)
            logger.info(f"Removed old checkpoint: {old}")

    def _make_gpu_accumulator(self):
        """Creates a new set of GPU tensors for metric accumulation — no CPU sync."""
        z = lambda: torch.zeros((), device=self.device)
        return {
            'loss_sum': z(),
            'td_sum': z(),
            'td_sq_sum': z(),
            'td_max': z(),
            'q_mean_sum': z(),
            'q_max_sum': z(),
            'q_min_sum': z(),
            'grad_norm_sum': z(),
            'q_spread_sum': z(),
            'action_entropy_sum': z(),
            'q_hold_sum': z(),
            'actions_total': torch.zeros(self._num_actions, dtype=torch.long, device=self.device),
        }

    def _snapshot_gpu_accumulator(self):
        """Snapshot of accumulator — one GPU→CPU sync, WITHOUT reset."""
        n = self._gpu_acc_steps
        if n == 0:
            return {
                'steps_accumulated': 0,
                'loss_sum': 0.0, 'td_error_sum': 0.0, 'td_error_sq_sum': 0.0,
                'td_error_max': 0.0, 'q_mean_sum': 0.0, 'q_max_sum': 0.0,
                'q_min_sum': 0.0, 'grad_norm_sum': 0.0, 'q_spread_sum': 0.0,
                'action_entropy_sum': 0.0, 'q_hold_sum': 0.0,
                'actions_total': [0] * self._num_actions,
            }
        scalars = torch.stack([
            self._gpu_acc['loss_sum'],       self._gpu_acc['td_sum'],
            self._gpu_acc['td_sq_sum'],      self._gpu_acc['td_max'],
            self._gpu_acc['q_mean_sum'],     self._gpu_acc['q_max_sum'],
            self._gpu_acc['q_min_sum'],      self._gpu_acc['grad_norm_sum'],
            self._gpu_acc['q_spread_sum'],   self._gpu_acc['action_entropy_sum'],
            self._gpu_acc['q_hold_sum'],
        ]).cpu().numpy()
        actions_total = self._gpu_acc['actions_total'].cpu().numpy().tolist()
        return {
            'steps_accumulated': n,
            'loss_sum':       float(scalars[0]),
            'td_error_sum':   float(scalars[1]),
            'td_error_sq_sum':float(scalars[2]),
            'td_error_max':   float(scalars[3]),
            'q_mean_sum':     float(scalars[4]),
            'q_max_sum':      float(scalars[5]),
            'q_min_sum':      float(scalars[6]),
            'grad_norm_sum':  float(scalars[7]),
            'q_spread_sum':   float(scalars[8]),
            'action_entropy_sum': float(scalars[9]),
            'q_hold_sum':     float(scalars[10]),
            'actions_total':  actions_total,
        }

    def _sync_gpu_accumulator(self):
        """Sync + reset — called only in _log_tensorboard."""
        if self._gpu_acc_steps == 0:
            return None
        result = self._snapshot_gpu_accumulator()
        self._gpu_acc = self._make_gpu_accumulator()
        self._gpu_acc_steps = 0
        return result

    @property
    def _metrics_accumulator(self):
        """Backward-compat: external callers (training_report, get_metrics) expect a dict.
        Each read = one GPU→CPU sync. Do NOT call in the hot path."""
        return self._snapshot_gpu_accumulator()

    def _augment_with_quantiles(self, state, default_symbol=None):
        """Inject normalized prerolled QuantileNet features into a state dict.

        Reads symbol/timestamp from the state itself (preferred) or falls back
        to default_symbol (e.g. actor_id when state was built before symbol
        was available). Mutates the dict in place and returns it.
        Loader misses fall back to zeros (and warn once per symbol).
        """
        if not isinstance(state, dict):
            return state
        symbol = state.get('symbol') or default_symbol
        ts = state.get('timestamp')
        state['quantile_features'] = self.quantile_loader.lookup(symbol, ts)
        return state

    def add_experience(self, state, action, reward, next_state, done, action_mask=None, actor_id=None):
        # Augment state with quantile features here (one lookup per experience)
        # rather than at training-step time (which would repeat lookups for every
        # PER sample). actor_id doubles as the symbol for ONNX actors.
        self._augment_with_quantiles(state, default_symbol=actor_id)
        if next_state is not None:
            self._augment_with_quantiles(next_state, default_symbol=actor_id)

        # Do not block ZMQ thread — if queue is full, skip (safe with PER)
        try:
            self._experience_queue.put_nowait((state, action, reward, next_state, done, action_mask))
        except queue.Full:
            pass

    def _drain_experience_queue(self) -> int:
        """Moves experiences from queue to buffer using one batch_add instead of N add() calls."""
        experiences = []
        try:
            while True:
                experiences.append(self._experience_queue.get_nowait())
        except queue.Empty:
            pass
        if experiences:
            self.buffer.batch_add(experiences)
        return len(experiences)

    def add_actor_metrics(self, symbol, metrics):
        """Adds actor metrics to the TensorBoard accumulator."""
        if not self.enable_actor_metrics:
            return
            
        actor_data = self._actor_metrics[symbol]
        
        # Transactions
        if 'transactions' in metrics:
            actor_data['transactions'] += metrics['transactions']
        
        # Episode PnL

        if 'episode_pnl' in metrics:
            pnl = metrics['episode_pnl']
            actor_data['episode_pnl_sum'] += pnl
            # episode_count incremented from episode_length below (avoid double-count)
            if pnl > 0:
                actor_data['wins'] += 1
                actor_data['current_consecutive_losses'] = 0
            elif pnl < 0:
                actor_data['losses'] += 1
                actor_data['current_consecutive_losses'] += 1
        
        # Total PnL
        if 'total_pnl' in metrics:
            actor_data['total_pnl'] = metrics['total_pnl']
        
        # Total trades
        if 'total_trades' in metrics:
            actor_data['total_trades'] = metrics['total_trades']
        
        # Win rate tracking (legacy binary format)
        if 'win' in metrics and metrics['win']:
            actor_data['wins'] += 1
        if 'loss' in metrics and metrics['loss']:
            actor_data['losses'] += 1

        # Episode metrics — sent by actor in the last entry of the batch
        if 'episode_length' in metrics:
            actor_data['episode_length_sum']   += metrics['episode_length']
            actor_data['episode_length_count'] += 1
            actor_data['episode_count']        += 1
        if 'episode_trades' in metrics:
            actor_data['transactions'] += metrics['episode_trades']
        if 'episode_wins' in metrics:
            actor_data['episode_trade_wins']   += metrics['episode_wins']
        if 'episode_losses' in metrics:
            actor_data['episode_trade_losses'] += metrics['episode_losses']
        if 'episode_max_consec_losses' in metrics:
            actor_data['max_consecutive_losses'] = max(
                actor_data['max_consecutive_losses'],
                metrics['episode_max_consec_losses']
            )

        # Extended trade statistics
        if 'episode_profit_sum' in metrics:
            actor_data['profit_sum']      += metrics['episode_profit_sum']
        if 'episode_loss_sum' in metrics:
            actor_data['loss_sum']        += metrics['episode_loss_sum']
        if 'episode_win_hold_steps' in metrics:
            actor_data['win_hold_steps']  += metrics['episode_win_hold_steps']
        if 'episode_loss_hold_steps' in metrics:
            actor_data['loss_hold_steps'] += metrics['episode_loss_hold_steps']
        if 'episode_gap_steps' in metrics:
            actor_data['gap_steps_deque'].extend(metrics['episode_gap_steps'])
        if 'episode_mfe_sum' in metrics:
            actor_data['mfe_sum']         += metrics['episode_mfe_sum']
        if 'episode_long_opens' in metrics:
            actor_data['long_opens']      += metrics['episode_long_opens']
        if 'episode_short_opens' in metrics:
            actor_data['short_opens']     += metrics['episode_short_opens']
        if 'episode_action_long' in metrics:
            actor_data['action_long']     += metrics['episode_action_long']
        if 'episode_action_short' in metrics:
            actor_data['action_short']    += metrics['episode_action_short']
        if 'episode_action_hold' in metrics:
            actor_data['action_hold']     += metrics['episode_action_hold']
        if 'episode_action_close' in metrics:
            actor_data['action_close']    += metrics['episode_action_close']

        # Episode PnL deque — for Sharpe/Sortino/MaxDrawdown
        if 'episode_pnl' in metrics:
            actor_data['episode_pnl_deque'].append(metrics['episode_pnl'])

    def _update_epsilon(self):
        total_decay_steps = self.epsilon_decay_fraction * self.config['training']['buffer_capacity']
        decay_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.step_count / max(total_decay_steps, 1))
        )

        if self._epsilon_reset_end_step is not None:
            if self.step_count < self._epsilon_reset_end_step:
                # Active reset — hold elevated epsilon
                self.epsilon = self._epsilon_reset_target
            else:
                # Reset expired — return to decay curve and clear state
                logger.info(f'[Trainer] Epsilon reset ended at step {self.step_count}, resuming decay ε={decay_epsilon:.4f}')
                self._epsilon_reset_end_step = None
                self._epsilon_reset_target   = None
                self.epsilon = decay_epsilon
        else:
            self.epsilon = decay_epsilon

        if self.lr_scheduler is not None and decay_epsilon > self.epsilon_end:
            self.lr_scheduler.step()

    def trigger_epsilon_reset(self, target: float | None = None, duration: int | None = None) -> None:
        """Temporarily raise epsilon to `target` for `duration` training steps.
        After expiry epsilon returns automatically to the current decay curve value.
        Default parameters come from config [training] epsilon_reset_target / epsilon_reset_duration.
        """
        cfg = self.config['training']
        target   = target   if target   is not None else cfg.get('epsilon_reset_target',   0.8)
        duration = duration if duration is not None else cfg.get('epsilon_reset_duration', 5000)

        old_epsilon = self.epsilon
        self._epsilon_reset_target   = float(target)
        self._epsilon_reset_end_step = self.step_count + int(duration)
        self.epsilon = self._epsilon_reset_target
        logger.info(
            f'[Trainer] Epsilon reset triggered: ε {old_epsilon:.4f} → {target:.4f} '
            f'for {duration:,} steps (ends at step {self._epsilon_reset_end_step:,})'
        )

    @property
    def dynamic_loss_scale(self) -> float:
        """loss_scale coupled to epsilon: 1.0 at epsilon_start → loss_scale_max at epsilon_end."""
        ls_max = self.config['reward'].get('loss_scale', 1.4)
        progress = (self.epsilon_start - self.epsilon) / max(self.epsilon_start - self.epsilon_end, 1e-8)
        progress = max(0.0, min(1.0, progress))
        return 1.0 + (ls_max - 1.0) * progress

    def _log_tensorboard(self, force=False):
        """Logs to TensorBoard if the time interval has elapsed."""
        current_time = time_module.time()

        # Check whether it's time to log scalar metrics
        if not force and (current_time - self._last_tb_log_time) < self.log_interval_sec:
            return

        # Check whether it's time for histograms
        log_histograms = (current_time - self._last_histogram_log_time) >= self.histogram_interval_sec
        
        step = self.step_count
        acc = self._sync_gpu_accumulator()  # JEDEN sync GPU→CPU

        if acc is not None:
            # Compute averages
            n = acc['steps_accumulated']
            avg_loss = acc['loss_sum'] / n
            self.last_loss = avg_loss  # update float for main.py logger
            avg_td = acc['td_error_sum'] / n
            td_std = np.sqrt(max(0.0, acc['td_error_sq_sum'] / n - avg_td**2)) if n > 1 else 0
            avg_q = acc['q_mean_sum'] / n
            avg_q_max = acc['q_max_sum'] / n
            avg_q_min = acc['q_min_sum'] / n
            avg_grad = acc['grad_norm_sum'] / n
            
            # ---- Core metrics ----
            self.writer.add_scalar('Loss/train', avg_loss, step)
            self.writer.add_scalar('Epsilon', self.epsilon, step)
            self.writer.add_scalar('Buffer/size', len(self.buffer), step)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning/lr', current_lr, step)
            
            # ---- TD Error ----
            self.writer.add_scalar('TD_error/mean', avg_td, step)
            self.writer.add_scalar('TD_error/max_abs', acc['td_error_max'], step)
            self.writer.add_scalar('TD_error/std', td_std, step)
            
            # ---- Q-values ----
            self.writer.add_scalar('Q_values/mean', avg_q, step)
            self.writer.add_scalar('Q_values/max', avg_q_max, step)
            self.writer.add_scalar('Q_values/min', avg_q_min, step)
            
            # EMA Q-values
            if self._q_values_ema is None:
                self._q_values_ema = avg_q
            self._q_values_ema = self._q_ema_decay * self._q_values_ema + (1 - self._q_ema_decay) * avg_q
            self.writer.add_scalar('Q_values/ema', self._q_values_ema, step)
            
            # ---- Gradients ----
            self.writer.add_scalar('Gradients/total_norm', avg_grad, step)

            # ---- Debugger v2: skalary ----
            self.writer.add_scalar('RL/q_spread', acc['q_spread_sum'] / n, step)
            self.writer.add_scalar('RL/action_entropy_bits', acc['action_entropy_sum'] / n, step)
            
            # ---- Buffer ----
            buffer_capacity = self.config['training']['buffer_capacity']
            buffer_fill_ratio = len(self.buffer) / buffer_capacity
            self.writer.add_scalar('Buffer/fill_ratio', buffer_fill_ratio, step)
            
            # ---- PER stats ----
            if self.use_per:
                self.writer.add_scalar('PER/beta', self.buffer.beta, step)
                self.writer.add_scalar('PER/max_priority', self.buffer.max_priority, step)
                if self.buffer.size > 0:
                    leaf_start = self.buffer.tree.capacity - 1
                    leaves = self.buffer.tree.tree[leaf_start:leaf_start + self.buffer.size]
                    non_zero = leaves[leaves > 0]
                    if len(non_zero) > 0:
                        self.writer.add_scalar('PER/priority_mean', float(non_zero.mean()), step)
                        self.writer.add_scalar('PER/priority_max', float(non_zero.max()), step)
            
            # ---- Actions distribution (summed across batches) ----
            total_actions = sum(acc['actions_total'])
            if total_actions > 0:
                for a, count in enumerate(acc['actions_total']):
                    self.writer.add_scalar(f'Actions/count_{a}', count, step)
                    self.writer.add_scalar(f'Actions/ratio_{a}', count / total_actions, step)
            
            # ---- Actor metrics ----
            if self.enable_actor_metrics:
                for symbol, data in self._actor_metrics.items():
                    prefix = f'Actor/{symbol}'
                    
                    # Win rate
                    total = data['wins'] + data['losses']
                    if total > 0:
                        win_rate = data['wins'] / total
                        self.writer.add_scalar(f'{prefix}/win_rate', win_rate, step)
                    
                    # Profit factor (if data is available)
                    if 'avg_win' in data and 'avg_loss' in data:
                        if data['avg_loss'] != 0:
                            pf = data['avg_win'] / abs(data['avg_loss'])
                            self.writer.add_scalar(f'{prefix}/profit_factor', pf, step)
                    
                    # Avg PnL per episode
                    if data['episode_count'] > 0:
                        avg_pnl = data['episode_pnl_sum'] / data['episode_count']
                        self.writer.add_scalar(f'{prefix}/episode_pnl_avg', avg_pnl, step)
                    
                    # Max consecutive losses
                    self.writer.add_scalar(f'{prefix}/max_consecutive_losses', data['max_consecutive_losses'], step)
                    
                    # Transactions
                    self.writer.add_scalar(f'{prefix}/transactions', data['transactions'], step)
            
            logger.debug(f"[TensorBoard] Logged at step {step}, accumulated {n} steps")

            # ---- Diagnostics: JSON Lines + alerty ----
            diag_metrics = {
                'loss': avg_loss,
                'q_mean': avg_q,
                'q_spread': acc['q_spread_sum'] / n,
                'grad_norm': avg_grad,
                'td_mean': avg_td,
                'td_std': td_std,
                'action_ratios': [c / total_actions for c in acc['actions_total']] if total_actions > 0 else [0.0] * 4,
                'advantage_std': self._last_advantage_std,
                'reward_mean': self._last_reward_mean,
                'epsilon': self.epsilon,
            }
            self.metric_logger.log(step, diag_metrics)
            self.alert_system.check_and_alert(diag_metrics, step)

        # Accumulator reset done inside _sync_gpu_accumulator()
        self._last_tb_log_time = current_time
        
        # Histograms (less frequently)
        if log_histograms:
            self._log_histograms(step)
            self._last_histogram_log_time = current_time

    def _log_histograms(self, step):
        """Logs histograms of weights, Q-values, rewards, and TD errors."""
        # Network weights
        for name, param in self.main_network.named_parameters():
            short_name = name.replace('.weight', '_w').replace('.bias', '_b')
            self.writer.add_histogram(f'Weights/{short_name}', param.data, step)

        # Debugger v2: histograms from the last batch
        if self._last_q is not None:
            self.debugger.log_q_diagnostics(self._last_q, step)
        if self._last_rewards is not None:
            self.debugger.log_reward_stats(self._last_rewards, step)
            self._last_reward_mean = self._last_rewards.detach().cpu().mean().item()
        if self._last_td is not None:
            self.debugger.log_td_histogram(self._last_td, step)

        logger.debug(f"[TensorBoard] Histograms logged at step {step}")

    def train_step(self):
        # Fill phase: drain before checking size (otherwise queue grows, buffer stalls).
        # Training phase: drain after check — return to old position.
        if not self._logged_training_start:
            self._drain_experience_queue()

        buffer_size = len(self.buffer)

        if buffer_size < self.min_buffer_size:
            if buffer_size >= self.min_buffer_size * 0.5 and not self._logged_halfway:
                logger.info(f"[Trainer] Buffer at 50%: {buffer_size}/{self.min_buffer_size}")
                self._logged_halfway = True
            return None

        if not self._logged_training_start:
            logger.info(f"[Trainer] TRAINING STARTED - Buffer full at {buffer_size} experiences, resuming={self.step_count > 0}")
            self._logged_training_start = True

        # Training phase — drain here (old position)
        self._drain_experience_queue()

        _t0 = time_module.perf_counter()

        if self.use_per:
            states, actions, rewards, next_states, dones, action_masks, pos_features, next_pos_features, indices, is_weights = self.buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones, action_masks, pos_features, next_pos_features, indices = self.buffer.sample(self.batch_size)
            is_weights = None

        _t1 = time_module.perf_counter()

        # Debugger v2: check whether it's time for heavy metrics
        debug_now = (time_module.time() - self._last_debug_time) >= self.debugger.debug_interval_sec

        self.main_network.train()

        if debug_now:
            self.debugger.register_hooks(self.main_network)
            self.debugger.snapshot_weights(self.main_network)
            self.attention_monitor.register()

        state_tensors = [states[k] for k in sorted(states.keys())]
        next_state_tensors = [next_states[k] for k in sorted(next_states.keys())]

        # ── Forward main (with grad) ──────────────────────────────────────────
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            current_q = self.main_network(state_tensors, position_features=pos_features)
            current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        _t2 = time_module.perf_counter()

        # ── Forward Double DQN (no grad, with autocast) ──────────────────────
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            # Derive the next-state action mask from the ACTUAL next-state position
            # (is_long = pos[:,0], is_short = pos[:,1]), not by reconstructing it from
            # the current action. The action-based reconstruction is wrong for n-step
            # returns: next_state is s_{t+n}, whose position depends on the intermediate
            # actions, not just a_t (e.g. a HOLD whose n-step window crosses a later
            # open would be mis-masked as flat). next_pos_features always reflects
            # s_{t+n}, so a position there means HOLD/CLOSE are legal, otherwise
            # LONG/SHORT/HOLD are. When done=True the bootstrap is zeroed, so the
            # mask for terminal transitions is irrelevant.
            in_position = (next_pos_features[:, 0] + next_pos_features[:, 1]) > 0.5
            next_masks = torch.where(
                in_position.unsqueeze(1), self._mask_in_pos, self._mask_no_pos
            )

            # Double DQN: main network selects action, target network evaluates value
            self.main_network.eval()
            next_q_main = self.main_network(next_state_tensors, action_mask=next_masks, position_features=next_pos_features)
            self.main_network.train()
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = self.target_network(next_state_tensors, action_mask=next_masks, position_features=next_pos_features)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = (rewards + (1 - dones) * self.gamma_n * next_q_values).float()

        _t3 = time_module.perf_counter()

        # ── Loss (outside autocast — MSE in fp32 for stability) ──────────────
        current_q_values_fp32 = current_q_values.float()
        td_errors = targets - current_q_values_fp32
        loss_per_sample = self.loss_fn(current_q_values_fp32, targets)

        if is_weights is not None:
            loss = (loss_per_sample * is_weights).mean()
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # AMP: unscale before computing gradient norm and clipping
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            self.main_network.parameters(), 1.0
        )

        # .item() forces GPU→CPU sync; only when we actually need to log.
        if not torch.isfinite(pre_clip_norm):
            logger.warning(f"[Trainer] Gradient explosion at step {self.step_count}: norm={pre_clip_norm.item():.2f} — AMP scaler will skip update")

        # Log gradient flow AFTER unscale and clipping — values are actual
        if debug_now:
            self.debugger.log_gradient_flow(self.main_network, self.step_count)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if debug_now:
            self.debugger.log_dead_neurons(self.step_count)
            self.debugger.log_weight_updates(self.main_network, self.step_count)
            self.debugger.log_value_advantage(
                self.main_network._debug_value,
                self.main_network._debug_advantage,
                self.step_count,
            )
            attn_metrics = self.attention_monitor.log_diagnostics(self.step_count)
            self.attention_monitor._remove_hooks()
            if self.main_network._debug_advantage is not None:
                self._last_advantage_std = self.main_network._debug_advantage.detach().std().item()
            self._last_debug_time = time_module.time()

        _t4 = time_module.perf_counter()

        if self.use_per:
            self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        _t5 = time_module.perf_counter()

        # Timing diagnostics — log 5 times after training starts, then every 500 steps
        if self._timing_logged < 5 or self.step_count % 500 == 0:
            self._timing_logged += 1
            logger.info(
                f"[Timing] sample={(_t1-_t0)*1000:.1f}ms  fwd_main={(_t2-_t1)*1000:.1f}ms  "
                f"fwd_double={(_t3-_t2)*1000:.1f}ms  bwd={(_t4-_t3)*1000:.1f}ms  "
                f"per_update={(_t5-_t4)*1000:.1f}ms  total_hot={(_t5-_t0)*1000:.1f}ms"
            )

        self.step_count += 1
        self._update_epsilon()

        if self.use_per:
            beta_fraction = min(1.0, self.step_count / self.config['training']['buffer_capacity'])
            self.buffer.update_beta(beta_fraction)

        # ===== Accumulate metrics — all on GPU, no .item() or .cpu() calls =====
        acc = self._gpu_acc
        with torch.no_grad():
            q_det = current_q.detach()
            td_det = td_errors.detach()
            td_abs = td_det.abs()

            acc['loss_sum']    += loss.detach()
            acc['td_sum']      += td_det.mean()
            acc['td_sq_sum']   += (td_det * td_det).mean()
            acc['td_max']      = torch.maximum(acc['td_max'], td_abs.max())
            acc['q_mean_sum']  += q_det.mean()
            q_max_row = q_det.max(dim=1).values
            q_min_row = q_det.min(dim=1).values
            acc['q_max_sum']   += q_max_row.mean()
            acc['q_min_sum']   += q_min_row.mean()
            if torch.isfinite(pre_clip_norm):
                acc['grad_norm_sum'] += pre_clip_norm
            acc['q_spread_sum']  += (q_max_row - q_min_row).mean()

            probs = torch.softmax(q_det.float(), dim=1)
            entropy = -(probs * torch.log2(probs.clamp(min=1e-8))).sum(dim=1).mean()
            acc['action_entropy_sum'] += entropy
            acc['q_hold_sum'] += q_det[:, 2].mean()

            # Instead of a Python loop over actions.cpu().numpy() — bincount on GPU
            acc['actions_total'] += torch.bincount(actions, minlength=self._num_actions)

        self._gpu_acc_steps += 1

        # Keep the last batch for histograms (still on GPU, sync happens in _log_histograms)
        self._last_q = q_det
        self._last_rewards = rewards.detach()
        self._last_td = td_det

        # ---- Log to TensorBoard at the configured interval ----
        self._log_tensorboard()

        if self.step_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            now_sps = time_module.time()
            elapsed = now_sps - self._sps_time_anchor
            steps_delta = self.step_count - self._sps_step_anchor
            sps = steps_delta / elapsed if elapsed > 0 else 0.0
            self._sps_step_anchor = self.step_count
            self._sps_time_anchor = now_sps
            logger.info(f"[Trainer] Target network updated at step {self.step_count} | {sps:.1f} steps/sec")
            self._export_onnx()
            self.training_report.maybe_send(self.step_count, self)

        if self.step_count % self.checkpoint_interval == 0:
            self.save_checkpoint()

        now = time_module.time()
        if now - self._last_hourly_ckpt_time >= 4 * 3600:
            hourly_path = os.path.join('python', 'checkpoints', 'hourly_checkpoint.pt')
            self.save_checkpoint(path=hourly_path, include_buffer=True)
            self._last_hourly_ckpt_time = now

        return self.last_loss

    @torch.no_grad()
    def predict(self, state, action_mask=None):
        # Inference must run with dropout off, but predict() may be invoked from the
        # ZMQ thread concurrently with train_step(). Restore the previous mode on exit
        # so a stray prediction never leaves the shared network stuck in eval().
        was_training = self.main_network.training
        self.main_network.eval()
        try:
            tf_keys = [k for k in sorted(self.config['timeframes'].keys()) if self.config['timeframes'][k] > 0]
            state_tensors = []
            for key in tf_keys:
                tf_name = key.replace('candles_', '')
                if tf_name in state:
                    tensor = torch.tensor(state[tf_name], dtype=torch.float32).unsqueeze(0).to(self.device, non_blocking=True)
                else:
                    seq_len = self.config['timeframes'][key]
                    tensor = torch.zeros(1, seq_len, self.config['features']['num_features'], device=self.device)
                state_tensors.append(tensor)

            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.tensor([action_mask], dtype=torch.float32).to(self.device, non_blocking=True)

            pos_tensor = None
            if isinstance(state, dict):
                # Build the full position vector: 10 base + quantile features.
                self._augment_with_quantiles(state)
                from training.replay_buffer import _extract_pos_vector
                quantile_dim = self.position_dim - 10
                vec = _extract_pos_vector(state, self.position_dim, quantile_dim)
                pos_tensor = torch.from_numpy(vec).unsqueeze(0).to(self.device, non_blocking=True)

            q_values = self.main_network(state_tensors, action_mask=mask_tensor, position_features=pos_tensor)
            action = q_values.argmax(dim=1).item()

            return action, q_values.squeeze(0).cpu().numpy()
        finally:
            if was_training:
                self.main_network.train()

    @torch.no_grad()
    def predict_action(self, state, action_mask=None):
        if np.random.random() < self.epsilon:
            valid_actions = (
                [i for i, m in enumerate(action_mask) if m == 1]
                if action_mask is not None
                else list(range(self.config['model']['num_actions']))
            )
            if not valid_actions:
                return np.random.randint(0, self.config['model']['num_actions'])

            # Biased exploration: P(HOLD) = clamp(epsilon, hold_bias_min, hold_bias_max).
            # Early training (high epsilon) → lots of HOLD → longer episodes.
            # Late training (low epsilon) → more LONG/SHORT/CLOSE → intensive exploration.
            training_cfg = self.config['training']
            hold_bias_min = training_cfg.get('hold_bias_min', 0.3)
            hold_bias_max = training_cfg.get('hold_bias_max', 0.7)
            hold_prob = max(hold_bias_min, min(hold_bias_max, self.epsilon))

            HOLD = 2
            non_hold = [a for a in valid_actions if a != HOLD]
            if HOLD not in valid_actions or not non_hold:
                return int(np.random.choice(valid_actions))

            other_prob = (1.0 - hold_prob) / len(non_hold)
            weights = [hold_prob if a == HOLD else other_prob for a in valid_actions]
            total = sum(weights)
            weights = [w / total for w in weights]
            return int(np.random.choice(valid_actions, p=weights))

        action, _ = self.predict(state, action_mask)
        return action

    def _export_onnx(self):
        """Exports the network to ONNX format — called on every target network update.
        Actors load the file locally and infer without RPC (issue #28).
        """
        onnx_cfg = self.config.get('onnx', {})
        if not onnx_cfg.get('enabled', False):
            return

        import warnings

        export_path = onnx_cfg.get('export_path', 'python/checkpoints/model.onnx')
        tmp_path = export_path + '.tmp'

        try:
            import torch.onnx

            tf_keys = [k for k in sorted(self.config['timeframes'].keys())
                       if self.config['timeframes'][k] > 0]
            num_features = self.config['features']['num_features']

            # Wrapper flattens the list of TF tensors + pos_features + action_mask
            # into positional arguments — ONNX requires named scalar inputs.
            class _OnnxWrapper(torch.nn.Module):
                def __init__(self, model, n_tf):
                    super().__init__()
                    self.model = model
                    self.n_tf = n_tf

                def forward(self, *args):
                    tf_tensors   = list(args[:self.n_tf])
                    pos_features = args[self.n_tf]
                    action_mask  = args[self.n_tf + 1]
                    return self.model(tf_tensors,
                                      action_mask=action_mask,
                                      position_features=pos_features)

            wrapper = _OnnxWrapper(self.main_network, len(tf_keys))
            wrapper.eval()

            dummy_inputs = []
            for key in tf_keys:
                seq_len = self.config['timeframes'][key]
                dummy_inputs.append(torch.zeros(1, seq_len, num_features, device=self.device))
            dummy_inputs.append(torch.zeros(1, self.position_dim, device=self.device))  # pos_features (10 base + quantile)
            dummy_inputs.append(torch.ones(1, 4, device=self.device))    # action_mask

            input_names  = [f'tf_{i}' for i in range(len(tf_keys))] + ['pos_features', 'action_mask']
            output_names = ['q_values']
            dynamic_axes = {n: {0: 'batch'} for n in input_names + output_names}

            os.makedirs(os.path.dirname(export_path), exist_ok=True)

            with torch.no_grad():
                export_kwargs = dict(
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=14,
                    do_constant_folding=True,
                )
                # PyTorch ≥2.1: dynamo=False forces the legacy TorchScript exporter
                # (no onnxscript required); older versions don't have this parameter
                import inspect
                if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                    export_kwargs['dynamo'] = False

                # aten::_native_multi_head_attention (fused C++ kernel) is not exportable
                # to any ONNX opset. In PyTorch 2.4+ it's called directly even during tracing
                # when batch_first=True. Class-level monkey-patch forces the slow path
                # (F.multi_head_attention_forward with explicit attn_mask), which ONNX
                # understands as separate Linear + softmax operations.
                _orig_mha_forward = torch.nn.MultiheadAttention.forward

                def _mha_slow_forward(self, query, key, value,
                                      key_padding_mask=None, need_weights=True,
                                      attn_mask=None, **kwargs):
                    if attn_mask is None:
                        # Zero-mask is a mathematical no-op, but `attn_mask is not None`
                        # disables the `_native_multi_head_attention` fast-path condition.
                        tgt = query.size(1) if self.batch_first else query.size(0)
                        attn_mask = query.new_zeros(tgt, tgt)
                    return _orig_mha_forward(self, query, key, value,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=need_weights,
                                             attn_mask=attn_mask, **kwargs)

                torch.nn.MultiheadAttention.forward = _mha_slow_forward  # apply patch
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='Constant folding')
                        torch.onnx.export(wrapper, tuple(dummy_inputs), tmp_path, **export_kwargs)
                finally:
                    torch.nn.MultiheadAttention.forward = _orig_mha_forward

            # Atomic rename — actor never sees a partial file
            if os.path.exists(export_path):
                os.remove(export_path)
            os.rename(tmp_path, export_path)
            logger.info(f"[ONNX] Exported model to {export_path} (step {self.step_count})")

        except Exception as e:
            logger.warning(f"[ONNX] Export failed at step {self.step_count}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _build_buffer(self):
        per_cfg = self.config.get('per', {})
        if per_cfg.get('alpha', 0) > 0:
            if per_cfg.get('positive_ratio', 0.0) > 0:
                return DualPrioritizedBuffer(self.config)
            return PrioritizedReplayBuffer(self.config)
        return ReplayBuffer(self.config)

    def reset_training(self):
        """Resets model, buffer, and training state to zero. Invoked by /restart in Telegram."""
        logger.info('[Trainer] Resetting training — fresh weights, empty buffer, step=0')

        # Fresh network weights
        self.main_network = TradingDQN(self.config).to(self.device)
        self.target_network = TradingDQN(self.config).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        # New optimizer
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        self.buffer = self._build_buffer()
        self._experience_queue = queue.Queue(maxsize=50000)

        # Reset training state
        self.step_count   = 0
        self.epsilon      = self.epsilon_start
        self.last_loss    = 0.0
        self._paused      = False
        self._logged_training_start = False
        self._logged_halfway = False
        self._last_q = None
        self._last_rewards = None
        self._last_td = None
        self._last_reward_mean = None
        self._sps_step_anchor = 0
        self._sps_time_anchor = time_module.time()
        self._timing_logged   = 0
        self._last_debug_time = 0.0
        self._last_tb_log_time = time_module.time()
        self._last_histogram_log_time = time_module.time()
        self._last_hourly_ckpt_time = time_module.time()
        self._gpu_acc       = self._make_gpu_accumulator()
        self._gpu_acc_steps = 0
        self._q_values_ema  = None
        self._last_advantage_std = None
        self._last_health   = None
        self._actor_metrics.clear()

        logger.info('[Trainer] Reset complete — ready for fresh training')

    def finish_current_step(self):
        # Force final log before closing
        self._log_tensorboard(force=True)
        self.debugger.close()
        self.writer.close()

    def get_metrics(self):
        """Returns metrics for the monitoring system (ZMQ push)."""
        acc = self._metrics_accumulator

        # Compute averages from accumulator
        n = acc['steps_accumulated']
        avg_loss = acc['loss_sum'] / n if n > 0 else 0.0
        avg_td = acc['td_error_sum'] / n if n > 0 else 0.0
        
        metrics = {
            'step': self.step_count,
            'epsilon': self.epsilon,
            'loss': self.last_loss,
            'loss_avg': avg_loss,  # average from accumulator
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.config['training']['buffer_capacity'],
            'buffer_fill_ratio': len(self.buffer) / self.config['training']['buffer_capacity'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'target_update_interval': self.target_update_interval,
            'use_per': self.use_per,
            'log_interval_sec': self.log_interval_sec,
            'histogram_interval_sec': self.histogram_interval_sec,
            # Q-values EMA
            'q_values_ema': self._q_values_ema if self._q_values_ema is not None else 0.0,
            # TD error average from accumulator
            'td_error_avg': avg_td,
        }
        
        # Grad norm from the last GPU acc sync
        gpu = self._gpu_acc
        gpu_n = self._gpu_acc_steps
        metrics['grad_norm'] = float(gpu['grad_norm_sum'] / gpu_n) if gpu_n > 0 else 0.0

        # PER-specific metrics
        if self.use_per:
            metrics['per_alpha'] = self.config.get('per', {}).get('alpha', 0)
            if hasattr(self.buffer, 'get_beta'):
                metrics['per_beta'] = self.buffer.get_beta(self.step_count)
        
        # Actor metrics summary
        if self.enable_actor_metrics and self._actor_metrics:
            actor_summary = {}
            for symbol, data in self._actor_metrics.items():
                total = data['wins'] + data['losses']
                actor_summary[symbol] = {
                    'transactions': data['transactions'],
                    'win_rate': data['wins'] / total if total > 0 else 0,
                    'max_consecutive_losses': data['max_consecutive_losses'],
                    'episode_pnl_avg': data['episode_pnl_sum'] / data['episode_count'] if data['episode_count'] > 0 else 0,
                }
            metrics['actors'] = actor_summary
        
        return metrics