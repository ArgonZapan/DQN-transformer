import os
import sys
import logging
import random
import torch
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.config           import load_config, get_device
from python.data.loader      import build_windows
from python.training.trainer import Trainer, sort_windows_chronologically


def setup_logging(config: dict):
    log_cfg = config.get('logging', {})
    level   = getattr(logging, log_cfg.get('level', 'INFO'), logging.INFO)
    log_dir = os.path.join(os.path.dirname(__file__), log_cfg.get('log_dir', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                os.path.join(log_dir, 'learner.log'),
                maxBytes=log_cfg.get('max_file_size_mb', 10) * 1024 * 1024,
                backupCount=log_cfg.get('max_files', 5),
            ),
        ],
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )


def main():
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    device = get_device(config)
    logger.info(f'[QuantileNet] Device: {device}')

    seed = int(config['training'].get('seed', 0))
    if seed <= 0:
        seed = random.SystemRandom().randrange(1, 2**31)
    logger.info(f'[QuantileNet] Seed: {seed}')

    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config['training'].get('num_threads', 4))

    # Load data directly from CSV for all symbols
    symbols = [a['symbol'] for a in config['actors']]
    all_windows = []
    for symbol in symbols:
        windows = build_windows(symbol, config)
        all_windows.extend(windows)
    all_windows = sort_windows_chronologically(all_windows)

    logger.info(f'[QuantileNet] Total windows: {len(all_windows)}')

    if not all_windows:
        logger.error('[QuantileNet] No data loaded — check config and CSV files.')
        sys.exit(1)

    trainer = Trainer(config, device)
    trainer.train(all_windows)


if __name__ == '__main__':
    main()
