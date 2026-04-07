import os
import sys
import signal
import logging
import threading

from config import load_config
from training.trainer import Trainer
from server.zmq_server import ZMQServer
from monitoring.monitor_client import MonitorClient

logger = logging.getLogger('learner')


def setup_logging(config):
    log_cfg = config['logging']
    os.makedirs(log_cfg['log_dir'], exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_cfg['log_dir'], 'learner.log'),
        maxBytes=log_cfg['max_file_size_mb'] * 1024 * 1024,
        backupCount=log_cfg['max_files']
    )
    console = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_cfg['level']))
    root.addHandler(handler)
    root.addHandler(console)


def main():
    import logging.handlers

    config = load_config()
    setup_logging(config)

    logger.info("Starting Python Learner...")
    logger.info(f"Device: {config['learner']['device']}")

    trainer = Trainer(config)
    logger.info(f"Model initialized on {config['learner']['device']}")

    monitor = MonitorClient(config)
    try:
        monitor.connect()
    except Exception as e:
        logger.warning(f"Could not connect to monitoring: {e}")

    zmq_server = ZMQServer(config, trainer)

    def graceful_shutdown(signum, frame):
        logger.info("Received shutdown signal — saving checkpoint...")
        trainer.finish_current_step()
        os.makedirs(os.path.join('checkpoints'), exist_ok=True)
        trainer.save_checkpoint(os.path.join('checkpoints', 'shutdown_checkpoint.pt'))
        logger.info("Checkpoint saved. Shutting down.")
        zmq_server.close()
        monitor.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    def training_loop():
        push_interval = config['monitoring']['metrics_push_interval_sec']
        import time
        last_push = 0

        while zmq_server.running:
            loss = trainer.train_step()
            if loss is not None and time.time() - last_push > push_interval:
                metrics = trainer.get_metrics()
                monitor.send_metrics('learner', metrics)
                last_push = time.time()
            elif loss is None:
                time.sleep(0.1)

    zmq_server.bind()

    train_thread = threading.Thread(target=training_loop, daemon=True)
    train_thread.start()

    logger.info("Learner ready. Waiting for actor connections...")
    zmq_server.start()


if __name__ == '__main__':
    main()
