import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TBLogger:
    """Thin TensorBoard wrapper with time-gated logging."""

    def __init__(self, config: dict):
        tb_cfg   = config.get('tensorboard', {})
        log_dir  = os.path.join(os.path.dirname(__file__), '..', tb_cfg.get('log_dir', 'runs'))
        self.writer           = SummaryWriter(log_dir=log_dir)
        self.log_interval     = tb_cfg.get('log_interval_sec', 60)
        self.hist_interval    = tb_cfg.get('histogram_interval_sec', 300)
        self._last_log        = 0.0
        self._last_hist       = 0.0
        logger.info(f'[TensorBoard] Logging to {log_dir}')

    def log_scalars(self, scalars: dict, step: int):
        now = time.time()
        if now - self._last_log < self.log_interval:
            return
        self._last_log = now
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()

    def log_scalars_immediate(self, scalars: dict, step: int):
        """Log without time gating (e.g. epoch metrics)."""
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()

    def log_histograms(self, model, step: int):
        now = time.time()
        if now - self._last_hist < self.hist_interval:
            return
        self._last_hist = now
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'grad/{name}', param.grad, step)
            self.writer.add_histogram(f'weight/{name}', param.data, step)
        self.writer.flush()

    def close(self):
        self.writer.close()
