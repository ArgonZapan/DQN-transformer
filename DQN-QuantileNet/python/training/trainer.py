import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.tft              import QuantileNet
from ..training.dataset       import QuantileDataset, collate_fn
from ..training.losses        import combined_loss
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

        self.q_weight  = pred_cfg.get('quantile_loss_weight',  0.5)
        self.t_weight  = pred_cfg.get('threshold_loss_weight', 0.5)
        self.mode      = pred_cfg['mode']
        self.quantiles = torch.tensor(pred_cfg.get('quantiles', []), dtype=torch.float32).to(device)

        self.model     = QuantileNet(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_cfg['lr'])

        lr_sched = train_cfg.get('lr_scheduler', 'none')
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg['epochs'], eta_min=train_cfg.get('lr_min', 1e-5)
            ) if lr_sched == 'cosine' else None
        )

        if train_cfg.get('compile_model', False):
            self.model = torch.compile(self.model)

        self.tb = TBLogger(config)
        self._saved_ckpts: list[str] = []
        os.makedirs(self.ckpt_dir, exist_ok=True)

        resume = train_cfg.get('resume_from_checkpoint', '')
        if resume:
            self._load_checkpoint(resume)

        self.backtest_interval = train_cfg.get('backtest_every_n_epochs', 1)
        self._test_windows: list = []
        self._ohlcv_by_symbol: dict = {}
        self._base_tf_min: int = 1

    def train(self, windows: list):
        active_tfs  = self.model.active_tfs
        tf_seq_lens = self.model.tf_seq_lens
        num_workers = min(4, os.cpu_count() or 1)
        pin = self.device.type == 'cuda'

        # ── Walk-forward split ───────────────────────────────────────────────────
        folds, test_windows = build_walk_forward_folds(
            windows, self.train_months, self.val_months, self.test_months
        )
        if not folds:
            logger.error('[Trainer] No folds built — check walk_forward config.')
            return

        # ── IS dataset: all train blocks from all folds combined ─────────────────
        is_windows = []
        for f in folds:
            is_windows.extend(f['train'])
        is_windows = sort_windows_chronologically(is_windows)

        is_ds     = QuantileDataset(is_windows, self.cfg, active_tfs, tf_seq_lens)
        is_loader = DataLoader(is_ds, batch_size=self.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers,
                               pin_memory=pin)

        # ── OOS loaders: one per fold (val block) ────────────────────────────────
        oos_loaders: list[DataLoader] = []
        for f in folds:
            ds = QuantileDataset(f['val'], self.cfg, active_tfs, tf_seq_lens)
            oos_loaders.append(DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=collate_fn, num_workers=num_workers,
                                          pin_memory=pin))

        logger.info(
            f'[Trainer] Walk-forward: {len(folds)} fold(s) | '
            f'IS={len(is_windows)} windows | '
            f'OOS per fold: {[len(f["val"]) for f in folds]} | '
            f'Test={len(test_windows)}'
        )

        # ── Per-epoch backtest setup (reuses already-built test_windows) ─────────
        if self.backtest_interval > 0:
            self._test_windows = test_windows
            self._ohlcv_by_symbol, self._base_tf_min = load_ohlcv(self.cfg)
            logger.info(f'[Trainer] Backtest ready: {len(self._test_windows)} test windows.')

        # Use OOS[last fold] as the model-selection signal (closest to test)
        best_val   = float('inf')
        no_improve = 0
        global_step = 0
        epoch_history: list[dict] = []
        n_folds = len(folds)

        for epoch in range(1, self.epochs + 1):
            # ── IS training pass ─────────────────────────────────────────────────
            is_loss = self._run_epoch(is_loader, training=True, step=global_step)
            global_step += len(is_loader)

            # ── OOS evaluation (no grad) — one loss per fold ─────────────────────
            oos_losses = [
                self._run_epoch(loader, training=False, step=global_step)
                for loader in oos_loaders
            ]

            if self.scheduler:
                self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]['lr']

            # ── Logging ──────────────────────────────────────────────────────────
            oos_str = '  '.join(f'OOS[{i+1}]={v:.5f}' for i, v in enumerate(oos_losses))
            logger.info(
                f'Epoch {epoch}/{self.epochs} | IS={is_loss:.5f}  {oos_str} | lr={lr_now:.2e}'
            )

            tb_scalars = {'loss/IS': is_loss, 'lr': lr_now}
            for i, v in enumerate(oos_losses):
                tb_scalars[f'loss/OOS_fold{i+1}'] = v
            self.tb.log_scalars_immediate(tb_scalars, global_step)

            # ── Checkpoint & early stopping (based on last fold OOS) ─────────────
            val_loss  = oos_losses[-1]
            is_best   = val_loss < best_val

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

            if self.backtest_interval > 0 and epoch % self.backtest_interval == 0:
                self._run_epoch_backtest(epoch)

        self.tb.close()
        logger.info(f'[Trainer] Done. Best val loss (OOS last fold): {best_val:.5f}')

    def _run_epoch(self, loader: DataLoader, training: bool, step: int) -> float:
        self.model.train(training)
        total, n = 0.0, 0

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

                outputs = self.model(tf_inputs)
                loss, components = combined_loss(
                    outputs, q_label, t_label, self.quantiles, self.q_weight, self.t_weight)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total += components['loss/total']
                n     += 1

        if training:
            self.tb.log_histograms(self.model, step)

        return total / max(n, 1)

    @staticmethod
    def _print_epoch_table(history: list[dict], n_folds: int, max_epochs: int):
        ep_w   = max(5, len(str(max_epochs)) + 1)
        col_w  = 9  # width for each loss column

        # Header
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
        torch.save({'epoch': epoch, 'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_loss': val_loss, 'config': self.cfg}, path)
        logger.info(f'[Trainer] Saved: {name}')
        if tag != 'best':
            self._saved_ckpts.append(path)
            while len(self._saved_ckpts) > self.keep_n_ckpts:
                old = self._saved_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        logger.info(f'[Trainer] Resumed from: {path}')

    def _run_epoch_backtest(self, epoch: int):
        logger.info(f'[Trainer] Running backtest after epoch {epoch} on current model state...')
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            first_param = next(self.model.parameters(), None)
            if first_param is not None:
                logger.info(
                    f'[Trainer] Model fingerprint before backtest: '
                    f'mean={first_param.detach().float().mean().item():.8f} '
                    f'std={first_param.detach().float().std().item():.8f}'
                )
        results = run_inference(self.model, self._test_windows, self.cfg, self.device)
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
