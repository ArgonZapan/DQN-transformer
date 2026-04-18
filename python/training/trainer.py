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

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['learner']['device']

        training_cfg = config['training']
        self.gamma = training_cfg['gamma']
        self.return_mode = training_cfg.get('return_mode', 'mc').lower()
        n_step = training_cfg.get('n_step', 1)
        # gamma_n — mnożnik bootstrappingu w celach Bellmana
        # mc:    done=True wszędzie → (1-done)*gamma_n = 0 → gamma_n nieistotne
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

        # Mixed Precision (FP16) — Ampere GPU (3090) ma tensor cores; ~2x speedup
        # CPU nie obsługuje GradScaler — wyłączamy automatycznie
        self.use_amp = self.device != 'cpu' and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("[Trainer] Mixed precision (AMP) enabled")

        # torch.compile — kompiluje graf, +20-40% speedup. UWAGA: koliduje z hookami
        # z debuggera/attention_monitora. Domyślnie wyłączone, włącz przez config.
        compile_enabled = config.get('training', {}).get('compile_model', False)
        if compile_enabled and hasattr(torch, 'compile') and self.device != 'cpu':
            try:
                self.main_network = torch.compile(self.main_network, mode='default', dynamic=False)
                self.target_network = torch.compile(self.target_network, mode='default', dynamic=False)
                logger.info("[Trainer] torch.compile enabled (mode=default)")
            except Exception as e:
                logger.warning(f"[Trainer] torch.compile failed, falling back to eager: {e}")

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='none')

        # Prealokowane stałe maski dla Double DQN (wcześniej tworzone co krok)
        self._mask_in_pos = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=self.device)
        self._mask_no_pos = torch.tensor([1, 1, 1, 0], dtype=torch.float32, device=self.device)

        # Kolejka doświadczeń: ZMQ thread tylko enqueue (szybkie), training thread
        # drainuje na początku każdego train_step — eliminuje GIL contention
        self._experience_queue: queue.Queue = queue.Queue(maxsize=50000)

        per_cfg = config.get('per', {})
        if per_cfg.get('alpha', 0) > 0:
            if per_cfg.get('positive_ratio', 0.0) > 0:
                self.buffer = DualPrioritizedBuffer(config)
                logger.info(f"[Trainer] Using DualPrioritizedBuffer (positive_ratio={per_cfg['positive_ratio']})")
            else:
                self.buffer = PrioritizedReplayBuffer(config)
            self.use_per = True
        else:
            self.buffer = ReplayBuffer(config)
            self.use_per = False

        self.step_count = 0
        self.epsilon = self.epsilon_start
        self.last_loss = 0.0
        self._logged_training_start = False
        self._sps_step_anchor = 0
        self._sps_time_anchor = time_module.time()
        self._timing_logged = 0  # ile razy zalogowano timing diagnostics
        
        # TensorBoard setup
        tensorboard_cfg = config.get('tensorboard', {})
        tensorboard_dir = tensorboard_cfg.get('log_dir', 'runs')
        self.log_interval_sec = tensorboard_cfg.get('log_interval_sec', 10)
        self.histogram_interval_sec = tensorboard_cfg.get('histogram_interval_sec', 60)
        self.enable_actor_metrics = tensorboard_cfg.get('enable_actor_metrics', True)
        
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(tensorboard_dir, f"train_{run_name}"))
        logger.info(f"[Trainer] TensorBoard logging to {tensorboard_dir}/train_{run_name} (interval={self.log_interval_sec}s)")
        
        # Czas ostatniego logowania TensorBoard
        self._last_tb_log_time = time_module.time()
        self._last_histogram_log_time = time_module.time()
        self._last_hourly_ckpt_time = time_module.time()
        
        # Debugger v2
        self.debugger = TensorBoardDebugger(self.writer, config)
        self._last_debug_time = 0.0

        # Moduł diagnostyczny
        diag_dir = 'python/diagnostics'
        self.metric_logger = MetricLogger(os.path.join(diag_dir, 'metrics.jsonl'), flush_every=50)
        self.alert_system = AlertSystem(config)
        self.training_report = TrainingReport(config)
        # Attention monitor — hooki robią .cpu() na KAŻDY forward pass (GPU sync).
        # Rejestrujemy dynamicznie tylko w oknie debug_now, potem usuwamy.
        self.attention_monitor = AttentionMonitor(self.main_network, self.writer)
        self._last_advantage_std: float | None = None
        self._last_health: dict | None = None

        # Akumulatory metryk — GPU tensory, jeden sync na log_interval zamiast 8+/krok
        self._num_actions = config['model']['num_actions']
        self._gpu_acc = self._make_gpu_accumulator()
        self._gpu_acc_steps = 0
        
        # Metryki od actorów (per symbol)
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
            # Statystyki epizodów (od aktora przez ZMQ)
            'episode_length_sum': 0,
            'episode_length_count': 0,
            'episode_trade_wins': 0,
            'episode_trade_losses': 0,
            # Statystyki handlowe rozszerzone
            'profit_sum': 0.0,          # suma zyskownych transakcji
            'loss_sum': 0.0,            # suma stratnych transakcji (abs)
            'win_hold_steps': 0,        # łączne kroki trzymania zyskownych pozycji
            'loss_hold_steps': 0,       # łączne kroki trzymania stratnych pozycji
            'mfe_sum': 0.0,             # suma MFE (Maximum Favorable Excursion)
            'long_opens': 0,
            'short_opens': 0,
            'action_long': 0,
            'action_short': 0,
            'action_hold': 0,
            'action_close': 0,
            # Deque ostatnich 200 episode PnL — do Sharpe/Sortino/MaxDD
            'episode_pnl_deque': deque(maxlen=200),
            # Deque ostatnich 5000 gap_steps — do mediany/p10/p90 czasu między transakcjami
            'gap_steps_deque': deque(maxlen=5000),
        })
        
        # EMA dla Q-values
        self._q_values_ema = None
        self._q_ema_decay = 0.99
        
        # Histogram akcji (ostatni batch)
        self._action_counts = [0] * config['model']['num_actions']

        self._load_checkpoint_if_exists()
        # Ustaw anchor SPS po załadowaniu checkpointu (step_count może być != 0 przy resume)
        self._sps_step_anchor = self.step_count
        self._sps_time_anchor = time_module.time()
        logger.info(f"[Trainer] Init: batch={self.batch_size}, min_buf={self.min_buffer_size}, steps={self.step_count}, resumed={self.step_count > 0}")

    def _load_checkpoint_if_exists(self):
        resume = self.config['training'].get('resume_from_checkpoint', '')
        if resume and os.path.exists(resume):
            self._load_checkpoint(resume)
            return

        ckpt_dir = os.path.join('python', 'checkpoints')

        # Zbierz WSZYSTKIE checkpointy i wybierz ten z najwyższym step jako model
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

        # Buffer wczytaj z najnowszego pliku który go zawiera (shutdown > hourly)
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

    def _load_checkpoint(self, path, load_buffer=True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.main_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            logger.warning(f"[Checkpoint] Architecture mismatch — starting with fresh weights. ({e})")
            return
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.last_loss = checkpoint.get('loss', 0.0)
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

        payload = {
            'step': self.step_count,
            'model_state_dict': self.main_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.last_loss,
            'epsilon': self.epsilon,
        }
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
        """Tworzy nowy zestaw GPU tensorów do akumulacji metryk — bez sync CPU."""
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

    def _sync_gpu_accumulator(self):
        """JEDEN sync GPU→CPU — stackuje wszystkie skalary i zwraca dict float'ów.
        Reset akumulatora po synchronizacji."""
        if self._gpu_acc_steps == 0:
            return None
        n = self._gpu_acc_steps
        # Stack wszystkich 11 skalarów w jeden tensor → jeden transfer
        scalars = torch.stack([
            self._gpu_acc['loss_sum'],       self._gpu_acc['td_sum'],
            self._gpu_acc['td_sq_sum'],      self._gpu_acc['td_max'],
            self._gpu_acc['q_mean_sum'],     self._gpu_acc['q_max_sum'],
            self._gpu_acc['q_min_sum'],      self._gpu_acc['grad_norm_sum'],
            self._gpu_acc['q_spread_sum'],   self._gpu_acc['action_entropy_sum'],
            self._gpu_acc['q_hold_sum'],
        ]).cpu().numpy()
        actions_total = self._gpu_acc['actions_total'].cpu().numpy().tolist()
        result = {
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
        # Reset — nowe GPU tensory (stare mogą być cached przez stack)
        self._gpu_acc = self._make_gpu_accumulator()
        self._gpu_acc_steps = 0
        return result

    def add_experience(self, state, action, reward, next_state, done, action_mask=None, actor_id=None):
        # Nie blokuj ZMQ thread — jeśli kolejka pełna, pomijamy (bezpieczne przy PER)
        try:
            self._experience_queue.put_nowait((state, action, reward, next_state, done, action_mask))
        except queue.Full:
            pass

    def _drain_experience_queue(self) -> int:
        """Przenosi doświadczenia z kolejki do buffera jednym batch_add zamiast N add()."""
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
        """Dodaje metryki od aktora do akumulatora TensorBoard."""
        if not self.enable_actor_metrics:
            return
            
        actor_data = self._actor_metrics[symbol]
        
        # Transakcje
        if 'transactions' in metrics:
            actor_data['transactions'] += metrics['transactions']
        
        # Episode PnL
        if 'episode_pnl' in metrics:
            pnl = metrics['episode_pnl']
            actor_data['episode_pnl_sum'] += pnl
            # episode_count incremented from episode_length below (nie duplikuj)
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
        
        # Win rate tracking (stary format binarny)
        if 'win' in metrics and metrics['win']:
            actor_data['wins'] += 1
        if 'loss' in metrics and metrics['loss']:
            actor_data['losses'] += 1

        # Metryki epizodu — wysyłane przez aktora w ostatnim wpisie batcha
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

        # Rozszerzone statystyki handlowe
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

        # Deque PnL epizodów — do Sharpe/Sortino/MaxDrawdown
        if 'episode_pnl' in metrics:
            actor_data['episode_pnl_deque'].append(metrics['episode_pnl'])

    def _update_epsilon(self):
        total_decay_steps = self.epsilon_decay_fraction * self.config['training']['buffer_capacity']
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.step_count / max(total_decay_steps, 1))
        )

    def _log_tensorboard(self, force=False):
        """Loguje do TensorBoard jeśli minął interwał czasowy."""
        current_time = time_module.time()
        
        # Sprawdź czy czas na logowanie scalar metrics
        if not force and (current_time - self._last_tb_log_time) < self.log_interval_sec:
            return
            
        # Sprawdź czy czas na histogramy
        log_histograms = (current_time - self._last_histogram_log_time) >= self.histogram_interval_sec
        
        step = self.step_count
        acc = self._sync_gpu_accumulator()  # JEDEN sync GPU→CPU

        if acc is not None:
            # Oblicz średnie
            n = acc['steps_accumulated']
            avg_loss = acc['loss_sum'] / n
            self.last_loss = avg_loss  # aktualizuj float dla main.py logger
            avg_td = acc['td_error_sum'] / n
            td_std = np.sqrt(max(0.0, acc['td_error_sq_sum'] / n - avg_td**2)) if n > 1 else 0
            avg_q = acc['q_mean_sum'] / n
            avg_q_max = acc['q_max_sum'] / n
            avg_q_min = acc['q_min_sum'] / n
            avg_grad = acc['grad_norm_sum'] / n
            
            # ---- Podstawowe metryki ----
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
            
            # ---- Actions distribution (sumowane z batchy) ----
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
                    
                    # Profit factor (jeśli mamy dane)
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
            total_actions = sum(acc['actions_total'])
            diag_metrics = {
                'loss': avg_loss,
                'q_mean': avg_q,
                'q_spread': acc['q_spread_sum'] / n,
                'grad_norm': avg_grad,
                'td_mean': avg_td,
                'td_std': td_std,
                'action_ratios': [c / total_actions for c in acc['actions_total']] if total_actions > 0 else [0.0] * 4,
                'advantage_std': self._last_advantage_std,
                'reward_mean': getattr(self, '_last_reward_mean', None),
                'epsilon': self.epsilon,
            }
            self.metric_logger.log(step, diag_metrics)
            self.alert_system.check_and_alert(diag_metrics, step)

        # Reset akumulatora zrobiony w _sync_gpu_accumulator()
        self._last_tb_log_time = current_time
        
        # Histogramy (rzadziej)
        if log_histograms:
            self._log_histograms(step)
            self._last_histogram_log_time = current_time

    def _log_histograms(self, step):
        """Loguje histogramy wag, Q-values, nagród i błędów TD."""
        # Wagi sieci
        for name, param in self.main_network.named_parameters():
            short_name = name.replace('.weight', '_w').replace('.bias', '_b')
            self.writer.add_histogram(f'Weights/{short_name}', param.data, step)

        # Debugger v2: histogramy z ostatniego batcha
        if hasattr(self, '_last_q') and self._last_q is not None:
            self.debugger.log_q_diagnostics(self._last_q, step)
        if hasattr(self, '_last_rewards') and self._last_rewards is not None:
            self.debugger.log_reward_stats(self._last_rewards, step)
            self._last_reward_mean = self._last_rewards.detach().cpu().mean().item()
        if hasattr(self, '_last_td') and self._last_td is not None:
            self.debugger.log_td_histogram(self._last_td, step)

        logger.debug(f"[TensorBoard] Histograms logged at step {step}")

    def train_step(self):
        buffer_size = len(self.buffer)
        
        if buffer_size < self.min_buffer_size:
            if buffer_size >= self.min_buffer_size * 0.5 and not hasattr(self, '_logged_halfway'):
                logger.info(f"[Trainer] Buffer at 50%: {buffer_size}/{self.min_buffer_size}")
                self._logged_halfway = True
            return None

        if not self._logged_training_start:
            logger.info(f"[Trainer] TRAINING STARTED - Buffer full at {buffer_size} experiences, resuming={self.step_count > 0}")
            self._logged_training_start = True

        # Drain kolejki doświadczeń (ZMQ thread enqueue, tu robimy buffer.add)
        self._drain_experience_queue()

        _t0 = time_module.perf_counter()

        if self.use_per:
            states, actions, rewards, next_states, dones, action_masks, pos_features, next_pos_features, indices, is_weights = self.buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones, action_masks, pos_features, next_pos_features, indices = self.buffer.sample(self.batch_size)
            is_weights = None

        _t1 = time_module.perf_counter()

        # Debugger v2: sprawdź czy czas na ciężkie metryki
        debug_now = (time_module.time() - self._last_debug_time) >= self.debugger.debug_interval_sec

        self.main_network.train()

        if debug_now:
            self.debugger.register_hooks(self.main_network)
            self.debugger.snapshot_weights(self.main_network)
            self.attention_monitor.register()

        state_tensors = [states[k] for k in sorted(states.keys())]
        next_state_tensors = [next_states[k] for k in sorted(next_states.keys())]

        # ── Forward main (z grad) ──────────────────────────────────────────
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            current_q = self.main_network(state_tensors, position_features=pos_features)
            current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        _t2 = time_module.perf_counter()

        # ── Forward Double DQN (bez grad, z autocast) ──────────────────────
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            next_masks = action_masks.clone()
            next_masks[actions == 0] = self._mask_in_pos  # po LONG
            next_masks[actions == 1] = self._mask_in_pos  # po SHORT
            next_masks[actions == 3] = self._mask_no_pos  # po CLOSE

            # Double DQN: main network wybiera akcję, target network ocenia wartość
            self.main_network.eval()
            next_q_main = self.main_network(next_state_tensors, action_mask=next_masks, position_features=next_pos_features)
            self.main_network.train()
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = self.target_network(next_state_tensors, action_mask=next_masks, position_features=next_pos_features)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = (rewards + (1 - dones) * self.gamma_n * next_q_values).float()

        _t3 = time_module.perf_counter()

        # ── Loss (poza autocast — MSE w fp32 dla stabilności) ──────────────
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

        if debug_now:
            self.debugger.log_gradient_flow(self.main_network, self.step_count)

        # AMP: unscale przed obliczeniem normy gradientu i clippingiem
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        # Trzymaj normę jako tensor — sync CPU dopiero przy logowaniu (jeden batch sync)
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            self.main_network.parameters(), 1.0
        )

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

        # Timing diagnostics — loguj 5 razy po starcie treningu, potem co 500 kroków
        if self._timing_logged < 5 or self.step_count % 500 == 0:
            self._timing_logged += 1
            _t_total = time_module.perf_counter()
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

        # ===== Akumuluj metryki — wszystko na GPU, żadnych .item() ani .cpu() =====
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
            acc['grad_norm_sum'] += pre_clip_norm
            acc['q_spread_sum']  += (q_max_row - q_min_row).mean()

            q_masked = q_det.masked_fill(action_masks == 0, float('-inf'))
            probs = torch.softmax(q_masked, dim=1)
            acc['action_entropy_sum'] += -(probs * torch.log2(probs + 1e-8)).sum(dim=1).mean()
            acc['q_hold_sum'] += q_det[:, 2].mean()

            # Zamiast pętli Python nad actions.cpu().numpy() — bincount na GPU
            acc['actions_total'] += torch.bincount(actions, minlength=self._num_actions)

        self._gpu_acc_steps += 1

        # Zachowaj ostatni batch dla histogramów (wciąż na GPU, sync dopiero w _log_histograms)
        self._last_q = q_det
        self._last_rewards = rewards.detach()
        self._last_td = td_det

        # ---- Loguj do TensorBoard co określony interwał ----
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
        if now - self._last_hourly_ckpt_time >= 3600:
            hourly_path = os.path.join('python', 'checkpoints', 'hourly_checkpoint.pt')
            self.save_checkpoint(path=hourly_path, include_buffer=True)
            self._last_hourly_ckpt_time = now

        return self.last_loss

    @torch.no_grad()
    def predict(self, state, action_mask=None):
        self.main_network.eval()

        tf_keys = [k for k in sorted(self.config['timeframes'].keys()) if self.config['timeframes'][k] > 0]
        state_tensors = []
        for key in tf_keys:
            tf_name = key.replace('candles_', '')
            if tf_name in state:
                tensor = torch.tensor(state[tf_name], dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                seq_len = self.config['timeframes'][key]
                tensor = torch.zeros(1, seq_len, self.config['features']['num_features']).to(self.device)
            state_tensors.append(tensor)

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.tensor([action_mask], dtype=torch.float32).to(self.device)

        pos_data = state.get('position') if isinstance(state, dict) else None
        pos_tensor = torch.tensor([pos_data[:4]], dtype=torch.float32).to(self.device) if pos_data is not None else None

        q_values = self.main_network(state_tensors, action_mask=mask_tensor, position_features=pos_tensor)
        action = q_values.argmax(dim=1).item()

        return action, q_values.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_action(self, state, action_mask=None):
        if np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                if valid_actions:
                    return np.random.choice(valid_actions)
            return np.random.randint(0, self.config['model']['num_actions'])

        action, _ = self.predict(state, action_mask)
        return action

    def _export_onnx(self):
        """Eksportuje sieć do formatu ONNX — wywoływana przy każdym update target network.
        Aktorzy ładują plik lokalnie i wnioskują bez RPC (issue #28).
        """
        onnx_cfg = self.config.get('onnx', {})
        if not onnx_cfg.get('enabled', False):
            return

        export_path = onnx_cfg.get('export_path', 'python/checkpoints/model.onnx')
        tmp_path = export_path + '.tmp'

        try:
            import torch.onnx

            tf_keys = [k for k in sorted(self.config['timeframes'].keys())
                       if self.config['timeframes'][k] > 0]
            num_features = self.config['features']['num_features']

            # Wrapper spłaszcza listę tensorów TF + pos_features + action_mask
            # do pozycyjnych argumentów — ONNX wymaga nazwanych skalarnych wejść.
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
            dummy_inputs.append(torch.zeros(1, 4, device=self.device))   # pos_features
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
                # PyTorch ≥2.1: dynamo=False wymusza legacy TorchScript exporter
                # (nie wymaga onnxscript); starsze wersje nie mają tego parametru
                import inspect
                if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
                    export_kwargs['dynamo'] = False

                # aten::_native_multi_head_attention (fused C++ kernel) nie jest exportowalny
                # do żadnego opset ONNX. W PyTorch 2.4+ jest wywoływany bezpośrednio nawet
                # podczas tracingu gdy batch_first=True. Monkey-patch na poziomie klasy wymusza
                # slow path (F.multi_head_attention_forward z explicit attn_mask), który ONNX
                # rozumie jako oddzielne operacje Linear + softmax.
                _orig_mha_forward = torch.nn.MultiheadAttention.forward

                def _mha_slow_forward(self, query, key, value,
                                      key_padding_mask=None, need_weights=True,
                                      attn_mask=None, **kwargs):
                    if attn_mask is None:
                        # Zero-mask to no-op matematycznie, ale `attn_mask is not None`
                        # wyłącza warunek `_native_multi_head_attention` fast path.
                        tgt = query.size(1) if self.batch_first else query.size(0)
                        attn_mask = query.new_zeros(tgt, tgt)
                    return _orig_mha_forward(self, query, key, value,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=need_weights,
                                             attn_mask=attn_mask, **kwargs)

                torch.nn.MultiheadAttention.forward = _mha_slow_forward
                try:
                    torch.onnx.export(wrapper, tuple(dummy_inputs), tmp_path, **export_kwargs)
                finally:
                    torch.nn.MultiheadAttention.forward = _orig_mha_forward

            # Atomowy rename — aktor nigdy nie widzi częściowego pliku
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

    def finish_current_step(self):
        # Force final log before closing
        self._log_tensorboard(force=True)
        self.debugger.close()
        self.writer.close()

    def get_metrics(self):
        """Zwraca metryki dla systemu monitoringu (ZMQ push)."""
        acc = self._metrics_accumulator
        
        # Oblicz średnie z akumulatora
        n = acc['steps_accumulated']
        avg_loss = acc['loss_sum'] / n if n > 0 else 0.0
        avg_td = acc['td_error_sum'] / n if n > 0 else 0.0
        
        metrics = {
            'step': self.step_count,
            'epsilon': self.epsilon,
            'loss': self.last_loss,
            'loss_avg': avg_loss,  # Średnia z akumulatora
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
            # Akcje z ostatniego batcha
            'action_counts': self._action_counts.copy(),
            # TD error avg z akumulatora
            'td_error_avg': avg_td,
        }
        
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