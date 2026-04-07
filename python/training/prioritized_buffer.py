import numpy as np
import torch


class SumTree:
    """
    SumTree — drzewo binarne do efektywnego samplowania z rozkładem priorytetów.
    Liście przechowują priorytety, węzły wewnętrzne sumę potomków.
    Operacja sample i update w O(log n) zamiast O(n).
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority):
        idx = self.data_pointer + self.capacity - 1
        self.update(self.data_pointer, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, data_idx, priority):
        idx = data_idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    def min_priority(self):
        leaf_start = self.capacity - 1
        leaves = self.tree[leaf_start:leaf_start + self.capacity]
        non_zero = leaves[leaves > 0]
        if len(non_zero) == 0:
            return 1.0
        return non_zero.min()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay z SumTree.
    Doświadczenia z większym TD error są samplowane częściej.
    Importance Sampling weights korygują bias.
    """

    def __init__(self, config):
        self.capacity = config['training']['buffer_capacity']
        self.num_features = config['features']['num_features']
        self.num_actions = config['model']['num_actions']
        self.device = config['learner']['device']

        per_cfg = config['per']
        self.alpha = per_cfg['alpha']
        self.beta_start = per_cfg['beta_start']
        self.beta_end = per_cfg['beta_end']
        self.per_epsilon = per_cfg['epsilon']

        self.beta = self.beta_start
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

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, action_mask=None, td_error=None):
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

        if td_error is not None:
            priority = (abs(td_error) + self.per_epsilon) ** self.alpha
        else:
            priority = self.max_priority ** self.alpha

        self.tree.add(priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            _, priority, data_idx = self.tree.get(s)
            indices.append(data_idx)
            priorities.append(priority)

        indices = np.array(indices)
        priorities = np.array(priorities, dtype=np.float64)

        sampling_probs = priorities / self.tree.total()
        sampling_probs = np.clip(sampling_probs, 1e-10, None)

        is_weights = (self.size * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        states_batch = {}
        next_states_batch = {}
        for key in self.timeframe_keys:
            states_batch[key] = self.states[key][indices].to(self.device, non_blocking=True)
            next_states_batch[key] = self.next_states[key][indices].to(self.device, non_blocking=True)

        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)
        action_masks = self.action_masks[indices].to(self.device, non_blocking=True)

        return states_batch, actions, rewards, next_states_batch, dones, action_masks, indices, is_weights

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.per_epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, fraction):
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * fraction

    def __len__(self):
        return self.size

    def is_ready(self, min_size):
        return self.size >= min_size
