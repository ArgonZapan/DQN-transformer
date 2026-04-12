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
from diagnostics.health_runner import HealthRunner
from diagnostics.baseline_comparator import BaselineComparator

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

    health_runner = HealthRunner(trainer, check_interval_sec=3600)
    baseline_comparator = BaselineComparator(config, config['learner']['device'])
    _baseline_done = baseline_comparator.computed   # True jeśli załadowano z pliku

    monitor = MonitorClient(config)
    try:
        monitor.connect()
    except Exception as e:
        logger.warning(f"Could not connect to monitoring: {e}")

    zmq_server = ZMQServer(config, trainer)

    # Ścieżka absolutna — niezależna od CWD
    shutdown_flag = os.path.abspath(os.path.join(os.path.dirname(__file__), 'shutdown.flag'))
    if os.path.exists(shutdown_flag):
        os.remove(shutdown_flag)

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
        # Usuń flagę po zapisie — stop.bat czeka aż flaga zniknie
        if os.path.exists(shutdown_flag):
            os.remove(shutdown_flag)
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    def training_loop():
        nonlocal _baseline_done
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
                    trainer.alert_system.notify_training_started(trainer.step_count, len(trainer.buffer))
                    training_started = True

                # Baseline — raz po zapełnieniu buffera
                if not _baseline_done:
                    try:
                        baseline_comparator.compute(trainer.buffer)
                    except Exception as e:
                        logger.warning(f"[Baseline] Failed: {e}")
                    _baseline_done = True

                # Hourly health check
                health_runner.maybe_run()

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
            if os.path.exists(shutdown_flag):
                logger.info("[Learner] Shutdown flag detected — saving checkpoint...")
                graceful_shutdown(None, None)  # usuwa flagę po zapisie, potem sys.exit
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        graceful_shutdown(None, None)


if __name__ == '__main__':
    main()
