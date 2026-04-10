import os
import sys
import signal
import logging
import logging.handlers
import threading
import time
import traceback

from config import load_config
from training.trainer import Trainer
from server.zmq_server import ZMQServer
from monitoring.monitor_client import MonitorClient

logger = logging.getLogger('learner')


def setup_logging(config):
    log_cfg = config['logging']
    log_dir = log_cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'learner.log')
    if os.path.exists(log_file):
        os.remove(log_file)

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg['max_file_size_mb'] * 1024 * 1024,
        backupCount=log_cfg['max_files'],
        encoding='utf-8',
    )
    console = logging.StreamHandler()
    console.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_cfg['level'].upper(), logging.INFO))
    root.addHandler(handler)
    root.addHandler(console)


def main():
    config = load_config()
    setup_logging(config)

    logger.info("Starting Python Learner...")
    device = config['learner']['device']
    logger.info(f"Device: {device}")

    if device == 'cuda':
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    trainer = Trainer(config)
    logger.info(f"Model initialized on {device}")

    monitor = MonitorClient(config)
    try:
        monitor.connect()
    except Exception as e:
        logger.warning(f"Could not connect to monitoring: {e}")

    zmq_server = ZMQServer(config, trainer)

    def graceful_shutdown(signum, frame):
        metrics = trainer.get_metrics()
        logger.info(f"[Learner] SHUTDOWN - Total steps: {metrics['step']}, "
                    f"final loss: {metrics['loss']:.6f}, epsilon: {metrics['epsilon']:.4f}")
        trainer.finish_current_step()
        shutdown_path = os.path.join('python', 'checkpoints', 'shutdown_checkpoint.pt')
        trainer.save_checkpoint(path=shutdown_path, include_buffer=True)
        logger.info("Checkpoint saved. Shutting down.")
        zmq_server.close()
        monitor.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    def training_loop():
        push_interval = config['monitoring']['metrics_push_interval_sec']
        last_push = 0.0
        last_buffer_log = 0.0
        training_started = False

        while zmq_server.running:
            try:
                loss = trainer.train_step()
            except Exception as e:
                logger.error(f"[Training] Exception in train_step: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)
                continue

            now = time.time()

            if loss is not None:
                if not training_started:
                    logger.info(f"[Training] STARTED: buffer={len(trainer.buffer)}, step={trainer.step_count}")
                    training_started = True
                if now - last_push >= push_interval:
                    metrics = trainer.get_metrics()
                    logger.info(f"[Training] step={metrics['step']}, loss={loss:.6f}, epsilon={metrics['epsilon']:.4f}")
                    monitor.send_metrics('learner', metrics)
                    last_push = now
            else:
                if now - last_buffer_log >= 5.0:
                    buf = len(trainer.buffer)
                    pct = 100 * buf / trainer.min_buffer_size
                    logger.info(f"[Training] Filling buffer: {buf}/{trainer.min_buffer_size} ({pct:.1f}%)")
                    last_buffer_log = now
                time.sleep(0.1)

    zmq_server.bind()

    zmq_thread = threading.Thread(target=zmq_server.start, daemon=True, name="ZMQThread")
    zmq_thread.start()
    logger.info(f"ZMQ thread started")

    train_thread = threading.Thread(target=training_loop, daemon=True, name="TrainingThread")
    train_thread.start()
    logger.info("Training thread started. Waiting for actor connections...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == '__main__':
    main()
