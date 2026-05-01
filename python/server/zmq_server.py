import logging
import msgpack
import zmq
import torch

from server.schemas import (
    validate_predict_request, validate_batch_request,
    build_step_response, build_predict_response,
)

logger = logging.getLogger(__name__)


class ZMQServer:
    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer
        self.host = config['learner']['host']
        self.port = config['learner']['port']

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)

        self.running = False

        self._actor_throttle_ms = int(config['training'].get('actor_throttle_ms', 0))

    def bind(self):
        address = f"tcp://*:{self.port}"
        self.socket.bind(address)
        logger.info(f"ZMQ Server bound to {address}")

    def start(self):
        self.running = True
        logger.info("ZMQ Server started, waiting for requests...")

        while self.running:
            try:
                message = self.socket.recv()
                data = msgpack.unpackb(message, raw=False)
                response = self._handle_request(data)
                packed = msgpack.packb(response)
                self.socket.send(packed)
                
                # Log every 10K messages (reduce spam)
                if hasattr(self, '_msg_count'):
                    self._msg_count += 1
                else:
                    self._msg_count = 1
                    
                if self._msg_count % 10000 == 0:
                    logger.info(f"[ZMQ] Processed {self._msg_count} messages, buffer={len(self.trainer.buffer)}")
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                try:
                    error_response = msgpack.packb({'error': str(e)})
                    self.socket.send(error_response)
                except Exception as send_err:
                    logger.error(f"Failed to send error response: {send_err}")

    def _handle_request(self, data):
        if isinstance(data, list):
            return self._handle_batch(data)
        elif 'action' in data and data['action'] is not None:
            return self._handle_step(data)
        else:
            return self._handle_predict(data)

    def _handle_step(self, data):
        # Validate all required fields are present
        required = ['state', 'action', 'reward', 'nextState', 'done']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing field in step request: {field}")
        
        state = data['state']
        action = data['action']
        reward = data['reward']
        next_state = data.get('nextState')
        done = data['done']
        action_mask = data.get('actionMask')

        if action is not None:
            self.trainer.add_experience(state, action, reward, next_state, done, action_mask, actor_id='single')

        if next_state is not None and not done:
            next_action = self.trainer.predict_action(next_state, action_mask)
        else:
            next_action = 2

        return build_step_response(next_action)

    def _handle_predict(self, data):
        validate_predict_request(data)
        state = data['state']
        action_mask = data.get('actionMask')

        action, q_values = self.trainer.predict(state, action_mask)
        return build_predict_response(action, q_values, epsilon=self.trainer.epsilon)

    def _handle_batch(self, batch):
        validate_batch_request(batch)

        # ONNX actors run inference locally and ignore the predict field in the response —
        # they only read epsilon. We do not call trainer.predict() per item (this used to
        # produce thousands of redundant size=1 GPU forward passes competing with training).
        for item in batch:
            actor_id = item['actorId']
            symbol = item.get('symbol', actor_id)
            action = item.get('action')
            reward = item.get('reward', 0)
            next_state = item.get('nextState')
            done = item.get('done', False)
            action_mask = item.get('actionMask')
            metrics = item.get('metrics', {})

            if action is not None:
                self.trainer.add_experience(
                    item['state'], action, reward, next_state, done, action_mask,
                    actor_id=actor_id
                )

            if metrics:
                self.trainer.add_actor_metrics(symbol, metrics)

        throttle_ms = 0
        if self._actor_throttle_ms > 0 and len(self.trainer.buffer) >= self.trainer.buffer.capacity:
            throttle_ms = self._actor_throttle_ms

        return {
            'epsilon':     float(self.trainer.epsilon),
            'loss_scale':  float(self.trainer.dynamic_loss_scale),
            'throttle_ms': throttle_ms,
        }

    def close(self):
        self.running = False
        logger.info("Closing ZMQ Server...")
        self.socket.setsockopt(zmq.LINGER, 500)  # wait up to 500ms for queued messages to send
        self.socket.close()
        self.context.term()
        logger.info("ZMQ Server closed.")
