import logging
import msgpack
import zmq
import torch

from server.schemas import (
    validate_step_request, validate_batch_request,
    build_step_response, build_predict_response, build_batch_response
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
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                try:
                    error_response = msgpack.packb({'error': str(e)})
                    self.socket.send(error_response)
                except Exception:
                    pass

    def _handle_request(self, data):
        if isinstance(data, list):
            return self._handle_batch(data)
        elif 'action' in data:
            return self._handle_step(data)
        else:
            return self._handle_predict(data)

    def _handle_step(self, data):
        validate_step_request(data)

        state = data['state']
        action = data['action']
        reward = data['reward']
        next_state = data.get('nextState')
        done = data['done']
        action_mask = data.get('actionMask')

        self.trainer.add_experience(state, action, reward, next_state, done, action_mask)

        if next_state is not None and not done:
            next_action = self.trainer.predict_action(next_state, action_mask)
        else:
            next_action = 2

        return build_step_response(next_action)

    def _handle_predict(self, data):
        validate_step_request(data) if 'action' in data else None
        state = data['state']
        action_mask = data.get('actionMask')

        action, q_values = self.trainer.predict(state, action_mask)
        return build_predict_response(action, q_values)

    def _handle_batch(self, batch):
        validate_batch_request(batch)

        results = []
        for item in batch:
            actor_id = item['actorId']
            state = item['state']
            action = item.get('action')
            reward = item.get('reward', 0)
            next_state = item.get('nextState')
            done = item.get('done', False)
            action_mask = item.get('actionMask')

            if action is not None:
                self.trainer.add_experience(state, action, reward, next_state, done, action_mask)

            pred_action, q_values = self.trainer.predict(state, action_mask)
            results.append({
                'actorId': actor_id,
                'action': pred_action,
                'qValues': q_values
            })

        return build_batch_response(results)

    def close(self):
        self.running = False
        logger.info("Closing ZMQ Server...")
        self.socket.close()
        self.context.term()
        logger.info("ZMQ Server closed.")
