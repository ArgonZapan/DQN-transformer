import math
import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from ..model.tft              import QuantileNet, upgrade_legacy_state_dict
from ..training.dataset       import QuantileDataset, GPUWindowDataset, collate_fn
from ..training.losses        import LossModule, compute_pos_freq
from ..training.walk_forward  import build_walk_forward_folds
from ..utils.tensorboard      import TBLogger
from ..backtest               import (load_ohlcv, run_inference, brute_force_tp_sl_top20,
                                      calibration_report, export_strategy_top100)

logger = logging.getLogger(__name__)


def sort_windows_chronologically(windows: list) -> list:
    """Sort windows globally by time so multi-symbol splits stay chronological."""
    return sorted(
        windows,
        key=lambda w: (int(w['current_close_time']), str(w.get('symbol', ''))),
    )


def _split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Param groups: weight_decay only on Linear/Conv weights; 0 for LayerNorm/biases.

    Standard recipe for transformer training — keeps norm/bias parameters
    unregularized to avoid biasing toward zero scale, which hurts attention.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith('.bias') or 'norm' in name.lower() or 'bn' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return groups


def _build_lr_lambda(warmup_steps: int, total_steps: int, lr_min_ratio: float):
    """Linear warmup → cosine decay per-step.

    Returns multiplier ∈ [lr_min_ratio, 1.0] applied to base LR by LambdaLR.
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps  = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return lr_lambda


class Trainer:
    def __init__(self, config: dict, device: torch.device):
        self.cfg    = config
        self.device = device
        train_cfg   = config['training']
        pred_cfg    = config['prediction']

        self.epochs        = train_cfg['epochs']
        self.patience      = train_cfg['patience']
        self.batch_size    = train_cfg['batch_size']
        self.ckpt_interval = train_cfg['checkpoint_interval']
        self.keep_n_ckpts  = train_cfg['keep_last_n_checkpoints']
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        self.ckpt_dir = os.path.join(base_ckpt, f'run_{run_ts}')

        self.train_months = train_cfg['train_months']
        self.val_months   = train_cfg['val_months']
        self.test_months  = train_cfg['test_months']

        self.mode      = pred_cfg['mode']

        # ── Model ────────────────────────────────────────────────────────────
        self.model = QuantileNet(config).to(device)

        # ── Loss module (owns quantiles, log_var params, pos_weight buffer) ──
        n_h = self.model.n_horizons
        n_t = self.model.n_thresholds
        n_d = self.model.n_directions
        self.loss_module = LossModule(config, n_horizons=n_h, n_thresholds=n_t, n_directions=n_d).to(device)

        # ── Optimizer (param groups + transformer-style betas/wd) ────────────
        wd       = float(train_cfg.get('weight_decay', 0.05))
        betas    = tuple(train_cfg.get('betas', [0.9, 0.95]))
        base_lr  = float(train_cfg['lr'])

        param_groups = _split_decay_param_groups(self.model, wd)
        # Loss-module learnable params (log_var) — never decayed.
        loss_params = [p for p in self.loss_module.parameters() if p.requires_grad]
        if loss_params:
            param_groups.append({'params': loss_params, 'weight_decay': 0.0})

        self.optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas)

        # ── LR scheduler: linear warmup + cosine, stepped per batch ──────────
        # `T_max`/total_steps is finalized in `train()` once the loader is built.
        self.lr_scheduler_kind = train_cfg.get('lr_scheduler', 'cosine')
        self.warmup_steps      = int(train_cfg.get('warmup_steps', 500))
        self.lr_min            = float(train_cfg.get('lr_min', 1e-7))
        self.scheduler         = None  # built inside `train()`

        # ── Mixed precision ──────────────────────────────────────────────────
        self.amp_dtype = str(train_cfg.get('amp_dtype', 'bf16')).lower()
        self._amp_enabled = self.amp_dtype in ('bf16', 'fp16') and device.type == 'cuda'
        self._amp_torch_dtype = {
            'bf16': torch.bfloat16, 'fp16': torch.float16,
        }.get(self.amp_dtype, torch.float32)
        # GradScaler only needed for fp16 (bf16 has fp32 dynamic range).
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if (self._amp_enabled and self.amp_dtype == 'fp16') else None
        )

        # ── Gradient clipping ────────────────────────────────────────────────
        self.grad_clip = float(train_cfg.get('grad_clip', 1.0))

        # ── EMA wrapper (for OOS eval / backtest) ────────────────────────────
        ema_decay = float(train_cfg.get('ema_decay', 0.0))
        self.ema_enabled = ema_decay > 0.0
        if self.ema_enabled:
            self.ema = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
                device=device,
            )
            logger.info(f'[Trainer] EMA enabled (decay={ema_decay}).')
        else:
            self.ema = None

        # ── Validation selector ──────────────────────────────────────────────
        self.val_selector = str(train_cfg.get('val_selector', 'mean_oos')).lower()

        if train_cfg.get('compile_model', False):
            self.model = torch.compile(self.model)

        self.tb = TBLogger(config)
        self._saved_ckpts: list[str] = []
        # Best checkpoints rotate independently of regular ones — keep only the
        # latest, since each is the full model+optimizer+EMA state (~hundreds of
        # MB) and val_loss is preserved in the filename for audit.
        self._saved_best_ckpts: list[str] = []
        os.makedirs(self.ckpt_dir, exist_ok=True)

        resume = train_cfg.get('resume_from_checkpoint', '')
        if resume:
            self._load_checkpoint(resume)

        self.backtest_interval      = int(train_cfg.get('backtest_every_n_epochs', 1))
        self.backtest_warmup_epochs = int(train_cfg.get('backtest_warmup_epochs', 0))
        self._test_windows: list = []
        self._ohlcv_by_symbol: dict = {}
        self._base_tf_min: int = 1

    # ─────────────────────────────────────────────────────────────────────────
    def train(self, windows: list):
        active_tfs  = self.model.active_tfs if not hasattr(self.model, '_orig_mod') else self.model._orig_mod.active_tfs
        tf_seq_lens = self.model.tf_seq_lens if not hasattr(self.model, '_orig_mod') else self.model._orig_mod.tf_seq_lens
        train_cfg = self.cfg.get('training', {})
        data_in_vram = bool(train_cfg.get('data_in_vram', False)) and self.device.type == 'cuda'
        cpu_cap = max(1, (os.cpu_count() or 1) // 2)
        num_workers = int(train_cfg.get('dataloader_workers', min(8, cpu_cap)))
        prefetch_factor = int(train_cfg.get('dataloader_prefetch_factor', 4))
        persistent_workers = bool(train_cfg.get('dataloader_persistent_workers', True))
        pin = self.device.type == 'cuda'

        # DataLoader kwargs that only apply when num_workers > 0.
        worker_kwargs = (
            {'persistent_workers': persistent_workers, 'prefetch_factor': prefetch_factor}
            if num_workers > 0 else {}
        )

        # ── Walk-forward split ───────────────────────────────────────────────
        folds, test_windows = build_walk_forward_folds(
            windows, self.train_months, self.val_months, self.test_months
        )
        if not folds:
            logger.error('[Trainer] No folds built — check walk_forward config.')
            return

        # ── IS dataset: all train blocks combined, augmented ─────────────────
        is_windows = []
        for f in folds:
            is_windows.extend(f['train'])
        is_windows = sort_windows_chronologically(is_windows)

        # Capture sizes up-front so we can free the source lists in the VRAM branch.
        n_is_windows  = len(is_windows)
        n_oos_per_fold = [len(f['val']) for f in folds]
        n_folds        = len(folds)

        if data_in_vram:
            logger.info('[Trainer] data_in_vram=true — staging IS+OOS windows to GPU.')
            is_loader = GPUWindowDataset(
                is_windows, self.cfg, active_tfs, tf_seq_lens, self.device,
                batch_size=self.batch_size, shuffle=True, train_augment=True,
            )
            oos_loaders = [
                GPUWindowDataset(
                    f['val'], self.cfg, active_tfs, tf_seq_lens, self.device,
                    batch_size=self.batch_size, shuffle=False, train_augment=False,
                )
                for f in folds
            ]
            # Drop CPU-side window refs now that everything's staged to GPU —
            # otherwise we'd pay 2× memory (RAM + VRAM) for the entire dataset.
            # `test_windows` stays alive for per-epoch backtest; `windows`
            # (function arg) and `folds` train/val lists no longer needed.
            import gc
            del is_windows
            for f in folds:
                f['train'] = None
                f['val']   = None
            windows.clear()
            gc.collect()
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                logger.info(f'[Trainer] GPU memory after staging: {vram_mb:.1f} MB allocated.')
        else:
            is_ds = QuantileDataset(is_windows, self.cfg, active_tfs, tf_seq_lens, train_augment=True)

            # Vol-stratified sampler is mutually exclusive with shuffle=True
            # (PyTorch raises if both are set). Sampler already randomizes order.
            if is_ds.sample_weights is not None:
                sampler = WeightedRandomSampler(
                    is_ds.sample_weights.tolist(),
                    num_samples=len(is_ds),
                    replacement=True,
                )
                is_loader = DataLoader(is_ds, batch_size=self.batch_size, sampler=sampler,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=pin, **worker_kwargs)
            else:
                is_loader = DataLoader(is_ds, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=pin, **worker_kwargs)

            # ── OOS loaders: one per fold (val block, no augmentation) ───────
            oos_loaders: list = []
            for f in folds:
                ds = QuantileDataset(f['val'], self.cfg, active_tfs, tf_seq_lens, train_augment=False)
                oos_loaders.append(DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                              collate_fn=collate_fn, num_workers=num_workers,
                                              pin_memory=pin, **worker_kwargs))

        logger.info(
            f'[Trainer] Walk-forward: {n_folds} fold(s) | '
            f'IS={n_is_windows} windows | '
            f'OOS per fold: {n_oos_per_fold} | '
            f'Test={len(test_windows)}'
        )

        # ── Build per-step scheduler now that we know steps/epoch ────────────
        if self.lr_scheduler_kind == 'cosine':
            total_steps = max(1, len(is_loader) * self.epochs)
            base_lr = self.optimizer.param_groups[0]['lr']
            lr_min_ratio = self.lr_min / base_lr if base_lr > 0 else 0.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=_build_lr_lambda(self.warmup_steps, total_steps, lr_min_ratio),
            )
            logger.info(
                f'[Trainer] LR schedule: warmup={self.warmup_steps} steps -> cosine '
                f'over {total_steps} total steps (base_lr={base_lr:.2e}, lr_min={self.lr_min:.2e}).'
            )

        # ── Compute pos_weight for BCE from training data ────────────────────
        if self.mode in ('threshold', 'both') and self.loss_module.bce_use_pos_weight:
            # Use the IS loader directly, capping at a reasonable number of batches
            # to keep startup fast on huge datasets.
            max_batches = min(len(is_loader), 200)
            pos_freq = compute_pos_freq(is_loader, self.device, max_batches=max_batches)
            if pos_freq is not None:
                self.loss_module.set_pos_weight(pos_freq)
                logger.info(
                    f'[Trainer] BCE pos_weight set from {max_batches} batches: '
                    f'pos_freq mean={pos_freq.mean().item():.3f} '
                    f'min={pos_freq.min().item():.3f} max={pos_freq.max().item():.3f}'
                )

        # ── Per-epoch backtest setup ─────────────────────────────────────────
        if self.backtest_interval > 0:
            self._test_windows = test_windows
            self._ohlcv_by_symbol, self._base_tf_min = load_ohlcv(self.cfg)
            logger.info(f'[Trainer] Backtest ready: {len(self._test_windows)} test windows.')

        best_val   = float('inf')
        no_improve = 0
        global_step = 0
        epoch_history: list[dict] = []

        for epoch in range(1, self.epochs + 1):
            # ── IS training pass ─────────────────────────────────────────────
            is_loss = self._run_epoch(is_loader, training=True, step=global_step)
            global_step += len(is_loader)

            # ── OOS evaluation (uses EMA weights when available) ─────────────
            eval_model = self.ema if self.ema_enabled else None
            oos_losses = [
                self._run_epoch(loader, training=False, step=global_step, eval_model=eval_model)
                for loader in oos_loaders
            ]

            lr_now = self.optimizer.param_groups[0]['lr']

            # ── Logging ──────────────────────────────────────────────────────
            oos_str = '  '.join(f'OOS[{i+1}]={v:.5f}' for i, v in enumerate(oos_losses))
            logger.info(
                f'Epoch {epoch}/{self.epochs} | IS={is_loss:.5f}  {oos_str} | lr={lr_now:.2e}'
            )

            tb_scalars = {'loss/IS': is_loss, 'lr': lr_now}
            for i, v in enumerate(oos_losses):
                tb_scalars[f'loss/OOS_fold{i+1}'] = v
            self.tb.log_scalars_immediate(tb_scalars, global_step)

            # ── Validation selector ──────────────────────────────────────────
            if self.val_selector == 'mean_oos':
                val_loss = float(sum(oos_losses) / max(len(oos_losses), 1))
            else:
                val_loss = float(oos_losses[-1])

            is_best = val_loss < best_val

            if epoch % self.ckpt_interval == 0:
                self._save_checkpoint(epoch, val_loss)

            if is_best:
                best_val   = val_loss
                no_improve = 0
                self._save_checkpoint(epoch, val_loss, tag='best')
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f'[Trainer] Early stopping at epoch {epoch}.')
                    break

            epoch_history.append({
                'epoch': epoch, 'is': is_loss, 'oos': oos_losses,
                'lr': lr_now, 'best': is_best,
            })
            self._print_epoch_table(epoch_history, n_folds, self.epochs)

            if (
                self.backtest_interval > 0
                and epoch > self.backtest_warmup_epochs
                and (epoch - self.backtest_warmup_epochs) % self.backtest_interval == 0
            ):
                self._run_epoch_backtest(epoch)

        self.tb.close()
        logger.info(f'[Trainer] Done. Best val loss ({self.val_selector}): {best_val:.5f}')

    # ─────────────────────────────────────────────────────────────────────────
    def _run_epoch(
        self, loader: DataLoader, training: bool, step: int,
        eval_model: nn.Module | None = None,
    ) -> float:
        """Single epoch over a loader.

        When `training=False` and `eval_model` is provided (e.g. EMA), the
        forward pass goes through that model instead of `self.model`.
        """
        if training:
            self.model.train()
            forward_model = self.model
        else:
            self.model.eval()
            forward_model = eval_model if eval_model is not None else self.model
            if hasattr(forward_model, 'eval'):
                forward_model.eval()

        # Accumulate loss as a GPU tensor and sync once at epoch end (avoids
        # per-batch .item() which forces cudaStreamSynchronize and stalls the
        # forward/backward pipeline).
        total = torch.zeros((), device=self.device)
        n = 0

        with torch.set_grad_enabled(training):
            for batch in loader:
                tf_inputs = {tf: t.to(self.device, non_blocking=True)
                             for tf, t in batch['tf'].items()}
                q_label = batch.get('quantile_label')
                t_label = batch.get('threshold_label')
                if q_label is not None:
                    q_label = q_label.to(self.device, non_blocking=True)
                if t_label is not None:
                    t_label = t_label.to(self.device, non_blocking=True)

                if self._amp_enabled:
                    with torch.amp.autocast('cuda', dtype=self._amp_torch_dtype):
                        outputs = forward_model(tf_inputs)
                        loss, components = self.loss_module(outputs, q_label, t_label)
                else:
                    outputs = forward_model(tf_inputs)
                    loss, components = self.loss_module(outputs, q_label, t_label)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        if self.grad_clip > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    if self.ema_enabled:
                        self.ema.update_parameters(self.model)

                total = total + components['loss/total'].detach()
                n     += 1

        if training:
            self.tb.log_histograms(self.model, step)

        return (total / max(n, 1)).item()

    @staticmethod
    def _print_epoch_table(history: list[dict], n_folds: int, max_epochs: int):
        ep_w   = max(5, len(str(max_epochs)) + 1)
        col_w  = 9

        header = f"{'Epoch':>{ep_w}} | {'IS':>{col_w}}"
        for i in range(n_folds):
            header += f" | {f'OOS[{i+1}]':>{col_w}}"
        header += f" | {'LR':>10}"
        sep = '-' * len(header)

        lines = [sep, header, sep]
        for row in history:
            best_marker = '*' if row.get('best') else ' '
            line = f"{str(row['epoch']) + best_marker:>{ep_w}} | {row['is']:>{col_w}.5f}"
            for v in row['oos']:
                line += f" | {v:>{col_w}.5f}"
            line += f" | {row['lr']:>10.2e}"
            lines.append(line)
        lines.append(sep)

        print('\n' + '\n'.join(lines), flush=True)

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = ''):
        name = f'ckpt_epoch{epoch}{"_" + tag if tag else ""}_val{val_loss:.4f}.pt'
        path = os.path.join(self.ckpt_dir, name)
        state = {
            'epoch':     epoch,
            'model':     self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss':      self.loss_module.state_dict(),
            'val_loss':  val_loss,
            'config':    self.cfg,
        }
        if self.ema_enabled:
            state['ema'] = self.ema.state_dict()
        torch.save(state, path)
        logger.info(f'[Trainer] Saved: {name}')
        if tag == 'best':
            self._saved_best_ckpts.append(path)
            while len(self._saved_best_ckpts) > 1:
                old = self._saved_best_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
        else:
            self._saved_ckpts.append(path)
            while len(self._saved_ckpts) > self.keep_n_ckpts:
                old = self._saved_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # Transparently upgrade checkpoints saved with the shared
        # Linear(1, d_model) feat_proj (exact functional equivalent).
        model_sd = upgrade_legacy_state_dict(self.model, ckpt['model'])
        self.model.load_state_dict(model_sd)
        if model_sd is not ckpt['model']:
            # Adam state (m, v) was tied to the old feat_proj shape — reset it.
            logger.info('[Trainer] Upgraded legacy feat_proj weights; optimizer state reset.')
        else:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'loss' in ckpt:
            self.loss_module.load_state_dict(ckpt['loss'])
        if 'ema' in ckpt and self.ema_enabled:
            self.ema.load_state_dict(upgrade_legacy_state_dict(self.ema, ckpt['ema']))
        logger.info(f'[Trainer] Resumed from: {path}')

    def _run_epoch_backtest(self, epoch: int):
        # Backtest uses EMA weights when available — they generalize better.
        bt_model = self.ema.module if self.ema_enabled else self.model
        logger.info(f'[Trainer] Running backtest after epoch {epoch} (model={"EMA" if self.ema_enabled else "raw"})...')
        was_training = self.model.training
        bt_model.eval()
        with torch.no_grad():
            first_param = next(bt_model.parameters(), None)
            if first_param is not None:
                logger.info(
                    f'[Trainer] Model fingerprint before backtest: '
                    f'mean={first_param.detach().float().mean().item():.8f} '
                    f'std={first_param.detach().float().std().item():.8f}'
                )
        results = run_inference(bt_model, self._test_windows, self.cfg, self.device)
        calibration_json = os.path.join(self.ckpt_dir, f'calibration_epoch{epoch}.json')
        calibration_report(results, self.cfg, save_path=calibration_json)
        bt_result = brute_force_tp_sl_top20(
            results, self._test_windows, self.cfg,
            self._ohlcv_by_symbol, self._base_tf_min,
        )
        if bt_result is not None:
            all_rows, meta = bt_result
            strategy_json = os.path.join(self.ckpt_dir, f'strategy_top100_epoch{epoch}.json')
            export_strategy_top100(all_rows, meta, strategy_json)
        if was_training:
            self.model.train()
