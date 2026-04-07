import torch
import numpy as np


class ReplayBuffer:
    """
    Pre-alokowany replay buffer z pinned memory.
    Zamiast listy obiektów Python, bufor to jeden duży blok pamięci per pole.
    Pinned memory umożliwia szybszy asynchroniczny transfer CPU→GPU przez DMA.
    """

    def __init__(self, config):
        self.capacity = config['training']['buffer_capacity']
        self.num_features = config['features']['num_features']
        self.num_actions = config['model']['num_actions']
        self.device = config['learner']['device']
        # Pinned memory — nie może być swapowana na dysk,
        # umożliwia szybszy transfer CPU→GPU przez DMA (tylko z GPU)
        self.use_pin = torch.cuda.is_available()

        tf_cfg = config['timeframes']
        self.timeframe_keys = sorted(tf_cfg.keys())
        self.timeframe_sizes = {k: tf_cfg[k] for k in self.timeframe_keys}

        self.states = {}
        self.next_states = {}
        for key in self.timeframe_keys:
            seq_len = self.timeframe_sizes[key]
            s = torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32)
            ns = torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32)
            self.states[key] = s.pin_memory() if self.use_pin else s
            self.next_states[key] = ns.pin_memory() if self.use_pin else ns

        a = torch.zeros(self.capacity, dtype=torch.long)
        r = torch.zeros(self.capacity, dtype=torch.float32)
        d = torch.zeros(self.capacity, dtype=torch.float32)
        m = torch.zeros(self.capacity, self.num_actions, dtype=torch.float32)
        self.actions = a.pin_memory() if self.use_pin else a
        self.rewards = r.pin_memory() if self.use_pin else r
        self.dones = d.pin_memory() if self.use_pin else d
        self.action_masks = m.pin_memory() if self.use_pin else m

        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, action_mask=None):
        idx = self.position

        for key in self.timeframe_keys:
            tf_name = key.replace('candles_', '')
            if tf_name in state:
                self.states[key][idx] = torch.tensor(state[tf_name], dtype=torch.float32)
            if next_state is not None and tf_name in next_state:
                self.next_states[key][idx] = torch.tensor(next_state[tf_name], dtype=torch.float32)

        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        if action_mask is not None:
            self.action_masks[idx] = torch.tensor(action_mask, dtype=torch.float32)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states_batch = {}
        next_states_batch = {}
        for key in self.timeframe_keys:
            # non_blocking=True — transfer asynchroniczny, CPU może kontynuować pracę
            states_batch[key] = self.states[key][indices].to(self.device, non_blocking=True)
            next_states_batch[key] = self.next_states[key][indices].to(self.device, non_blocking=True)

        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)
        action_masks = self.action_masks[indices].to(self.device, non_blocking=True)

        return states_batch, actions, rewards, next_states_batch, dones, action_masks, indices

    def __len__(self):
        return self.size

    def is_ready(self, min_size):
        return self.size >= min_size
