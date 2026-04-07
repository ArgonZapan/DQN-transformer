import os
import logging
import glob

import torch
import torch.nn as nn
import numpy as np

from model.network import TradingDQN
from training.replay_buffer import ReplayBuffer
from training.prioritized_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['learner']['device']

        training_cfg = config['training']
        self.gamma = training_cfg['gamma']
        self.lr = training_cfg['lr']
        self.batch_size = training_cfg['batch_size']
        self.min_buffer_size = training_cfg['min_buffer_size']
        self.target_update_interval = training_cfg['target_update_interval']
        self.epsilon_start = training_cfg['epsilon_start']
        self.epsilon_end = training_cfg['epsilon_end']
        self.epsilon_decay_fraction = training_cfg['epsilon_decay_fraction']
        self.checkpoint_interval = training_cfg['checkpoint_interval']
        self.keep_last_n = training_cfg['keep_last_n_checkpoints']
        self.seed = training_cfg['seed']

        if self.seed >= 0:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.main_network = TradingDQN(config).to(self.device)
        self.target_network = TradingDQN(config).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='none')

        per_cfg = config.get('per', {})
        if per_cfg.get('alpha', 0) > 0:
            self.buffer = PrioritizedReplayBuffer(config)
            self.use_per = True
        else:
            self.buffer = ReplayBuffer(config)
            self.use_per = False

        self.step_count = 0
        self.epsilon = self.epsilon_start
        self.last_loss = 0.0

        self._load_checkpoint_if_exists()

    def _load_checkpoint_if_exists(self):
        resume = self.config['training'].get('resume_from_checkpoint', '')
        if resume and os.path.exists(resume):
            self._load_checkpoint(resume)
            return

        shutdown_path = os.path.join('python', 'checkpoints', 'shutdown_checkpoint.pt')
        if os.path.exists(shutdown_path):
            logger.info(f"Resuming from shutdown checkpoint: {shutdown_path}")
            self._load_checkpoint(shutdown_path)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.main_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.last_loss = checkpoint.get('loss', 0.0)
        logger.info(f"Checkpoint loaded from {path} (step {self.step_count})")

    def save_checkpoint(self, path=None):
        if path is None:
            os.makedirs(os.path.join('python', 'checkpoints'), exist_ok=True)
            path = os.path.join('python', 'checkpoints', f'checkpoint_step_{self.step_count}.pt')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'step': self.step_count,
            'model_state_dict': self.main_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.last_loss,
            'epsilon': self.epsilon,
        }, path)

        logger.info(f"Checkpoint saved to {path}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        checkpoint_dir = os.path.join('python', 'checkpoints')
        pattern = os.path.join(checkpoint_dir, 'checkpoint_step_*.pt')
        checkpoints = sorted(glob.glob(pattern))
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            os.remove(old)
            logger.info(f"Removed old checkpoint: {old}")

    def add_experience(self, state, action, reward, next_state, done, action_mask=None):
        if self.use_per:
            self.buffer.add(state, action, reward, next_state, done, action_mask)
        else:
            self.buffer.add(state, action, reward, next_state, done, action_mask)

    def _update_epsilon(self):
        total_decay_steps = self.epsilon_decay_fraction * self.config['training']['buffer_capacity']
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.step_count / max(total_decay_steps, 1))
        )

    def train_step(self):
        if not self.buffer.is_ready(self.min_buffer_size):
            return None

        if self.use_per:
            states, actions, rewards, next_states, dones, action_masks, indices, is_weights = self.buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones, action_masks, indices = self.buffer.sample(self.batch_size)
            is_weights = None

        self.main_network.train()

        state_tensors = [states[k] for k in sorted(states.keys())]
        next_state_tensors = [next_states[k] for k in sorted(next_states.keys())]

        current_q = self.main_network(state_tensors)
        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: main network wybiera akcję, target network ocenia wartość
            next_q_main = self.main_network(next_state_tensors)
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = self.target_network(next_state_tensors)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = targets - current_q_values
        loss_per_sample = self.loss_fn(current_q_values, targets)

        if is_weights is not None:
            loss = (loss_per_sample * is_weights).mean()
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 10.0)
        self.optimizer.step()

        if self.use_per:
            self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.step_count += 1
        self.last_loss = loss.item()
        self._update_epsilon()

        if self.step_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            logger.info(f"Target network updated at step {self.step_count}")

        if self.step_count % self.checkpoint_interval == 0:
            self.save_checkpoint()

        return self.last_loss

    @torch.no_grad()
    def predict(self, state, action_mask=None):
        self.main_network.eval()

        tf_keys = sorted(self.config['timeframes'].keys())
        state_tensors = []
        for key in tf_keys:
            tf_name = key.replace('candles_', '')
            if tf_name in state:
                tensor = torch.tensor(state[tf_name], dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                seq_len = self.config['timeframes'][key]
                tensor = torch.zeros(1, seq_len, self.config['features']['num_features']).to(self.device)
            state_tensors.append(tensor)

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.tensor([action_mask], dtype=torch.float32).to(self.device)

        q_values = self.main_network(state_tensors, action_mask=mask_tensor)
        action = q_values.argmax(dim=1).item()

        return action, q_values.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_action(self, state, action_mask=None):
        if np.random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                if valid_actions:
                    return np.random.choice(valid_actions)
            return np.random.randint(0, self.config['model']['num_actions'])

        action, _ = self.predict(state, action_mask)
        return action

    def finish_current_step(self):
        pass

    def get_metrics(self):
        return {
            'step': self.step_count,
            'epsilon': self.epsilon,
            'loss': self.last_loss,
            'buffer_size': len(self.buffer),
        }
