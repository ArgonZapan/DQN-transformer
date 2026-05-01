"""Standalone live prediction runner.

Loads the latest checkpoint, runs inference on the most recent candle window
for each configured symbol, and writes python/live_predictions.json.

Run once:
    python -m python.live_predict

Run in a loop (every 15 minutes):
    python -m python.live_predict --loop --interval 900
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.config           import load_config, get_device
from python.backtest         import find_best_checkpoint, load_model
from python.server.live.loop import write_live_predictions, LIVE_JSON_PATH


def main():
    ap = argparse.ArgumentParser(description='QuantileNet live prediction writer')
    ap.add_argument('--checkpoint', default='', help='Path to .pt file (default: best in checkpoints/)')
    ap.add_argument('--loop', action='store_true', help='Run continuously')
    ap.add_argument('--interval', type=int, default=900, help='Seconds between writes in --loop mode (default: 900)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    device = get_device(config)

    ckpt_dir   = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint = args.checkpoint or find_best_checkpoint(ckpt_dir)
    logger.info(f'[LivePredict] Using checkpoint: {checkpoint}')

    model = load_model(config, device, checkpoint)
    model.eval()

    if args.loop:
        logger.info(f'[LivePredict] Loop mode: writing every {args.interval}s')
        while True:
            try:
                write_live_predictions(model, config, checkpoint, device)
            except Exception as e:
                logger.error(f'[LivePredict] Error: {e}')
            logger.info(f'[LivePredict] Next write in {args.interval}s — output: {LIVE_JSON_PATH}')
            time.sleep(args.interval)
    else:
        write_live_predictions(model, config, checkpoint, device)
        logger.info(f'[LivePredict] Done → {LIVE_JSON_PATH}')


if __name__ == '__main__':
    main()
