import os
import sys
import signal
import logging
import logging.handlers
import threading
import time

from config import load_config

# Reset logów przy starcie
def reset_logs(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'learner.log')
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"[Setup] Removed old log file: {log_file}")
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
    # Reset logów przy starcie
    reset_logs(config['logging']['log_dir'])
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
    logger.info(f"Model initialized on {config['learner']['device']}")

    monitor = MonitorClient(config)
    try:
        monitor.connect()
    except Exception as e:
        logger.warning(f"Could not connect to monitoring: {e}")

    zmq_server = ZMQServer(config, trainer)

    def graceful_shutdown(signum, frame):
        metrics = trainer.get_metrics()
        logger.info(f"[Learner] 🛑 SHUTDOWN - Total steps: {metrics['step']}, final loss: {metrics['loss']:.6f}, epsilon: {metrics['epsilon']:.4f}")
        logger.info("Saving checkpoint...")
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
        print("[Training] Thread started!")  # Debug - czy wątek w ogóle startuje
        import sys
        sys.stdout.flush()
        
        push_interval = config['monitoring']['metrics_push_interval_sec']
        import time
        last_push = 0
        last_buffer_log = 0
        training_started = False
        iteration = 0

        while zmq_server.running:
            iteration += 1
            if iteration <= 3 or iteration % 50 == 0:
                print(f"[Training] iteration={iteration}, buffer={len(trainer.buffer)}, running={zmq_server.running}")
                sys.stdout.flush()
            buffer_before = len(trainer.buffer)
            try:
                loss = trainer.train_step()
            except Exception as e:
                logger.error(f"[Training] Exception in train_step: {e}")
                time.sleep(1)
                continue
            
            if loss is not None:
                # First time training starts
                if not training_started:
                    logger.info(f"[Training] ✅ TRAINING STARTED: buffer={buffer_before}, step_count={trainer.step_count}")
                    for h in logging.root.handlers:
                        h.flush()
                    training_started = True
                # Regular metrics push
                if time.time() - last_push >= push_interval:
                    metrics = trainer.get_metrics()
                    logger.info(f"[Training] step={metrics['step']}, loss={loss:.6f}, epsilon={metrics['epsilon']:.4f}")
                    monitor.send_metrics('learner', metrics)
                    last_push = time.time()
            else:
                buffer_size = len(trainer.buffer)
                if time.time() - last_buffer_log >= 1.0:  # co 1 sekundę zamiast 2
                    logger.info(f"[Training] Buffer: {buffer_size}/{trainer.min_buffer_size} ({100*buffer_size/trainer.min_buffer_size:.1f}%)")
                    for h in logging.root.handlers:
                        h.flush()
                    last_buffer_log = time.time()
                time.sleep(0.1)

    zmq_server.bind()

    train_thread = threading.Thread(target=training_loop, daemon=True)
    train_thread.start()

    logger.info("Learner ready. Waiting for actor connections...")
    zmq_server.start()


if __name__ == '__main__':
    main()
