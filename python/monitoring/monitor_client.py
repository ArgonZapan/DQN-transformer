import logging
import json
import zmq

logger = logging.getLogger(__name__)


class MonitorClient:
    """Wysyła metryki do Monitoring Service przez ZeroMQ PUSH."""

    def __init__(self, config):
        self.host = config['monitoring']['host']
        self.port = config['monitoring']['metrics_pull_port']
        self.interval = config['monitoring']['metrics_push_interval_sec']

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)

        self._connected = False

    def connect(self):
        address = f"{self.host}:{self.port}"
        self.socket.connect(address)
        self._connected = True
        logger.info(f"Monitor client connected to {address}")

    def send_metrics(self, source, metrics):
        if not self._connected:
            return

        try:
            message = json.dumps({'source': source, 'metrics': metrics})
            self.socket.send_string(message, zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("Monitor push dropped (buffer full)")
        except Exception as e:
            logger.error(f"Monitor push error: {e}")

    def close(self):
        self.socket.close()
        self.context.term()
        self._connected = False
        logger.info("Monitor client closed.")
