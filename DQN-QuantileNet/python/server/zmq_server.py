import zmq
import msgpack
import logging
import threading

logger = logging.getLogger(__name__)


class ZMQServer:
    """ZMQ PULL server that receives data windows from Node.js data actors.

    Runs in a background thread. Calls trainer.add_batch() and trainer.mark_done()
    as messages arrive. When all expected symbols have signalled 'done', calls
    on_all_done() callback so the main thread can start training.
    """

    def __init__(self, config: dict, trainer, on_all_done):
        self.config       = config
        self.trainer      = trainer
        self.on_all_done  = on_all_done
        self._stop        = threading.Event()
        self._thread      = None

        self.expected_symbols = {a['symbol'] for a in config['actors']}
        self.received: dict[str, int] = {}

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name='zmq-server')
        self._thread.start()
        logger.info('[ZMQ] Server thread started.')

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        host = self.config['zmq']['host']
        port = self.config['zmq']['port']
        sock.bind(f'{host}:{port}')
        sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout to allow clean stop
        logger.info(f'[ZMQ] Bound to {host}:{port}')

        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue

                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception as e:
                    logger.warning(f'[ZMQ] Decode error: {e}')
                    continue

                msg_type = msg.get('type', '')
                symbol   = msg.get('symbol', 'unknown')

                if msg_type == 'data_batch':
                    windows = msg.get('windows', [])
                    self.trainer.add_batch(symbol, windows)
                    self.received[symbol] = self.received.get(symbol, 0) + len(windows)

                elif msg_type == 'done':
                    total = msg.get('total', 0)
                    logger.info(f'[ZMQ] {symbol} sent done signal. Total windows: {total}')
                    self.trainer.mark_done(symbol)

                    if self.trainer.all_actors_done(self.expected_symbols):
                        logger.info('[ZMQ] All actors done. Triggering training.')
                        self.on_all_done()
                        break

                else:
                    logger.warning(f'[ZMQ] Unknown message type: {msg_type}')

        finally:
            sock.close()
            ctx.term()
            logger.info('[ZMQ] Server stopped.')
