import torch
import numpy as np

from quantilenet import POSITION_BASE_DIM, extended_position_dim


def _extract_pos_vector(state, total_dim: int, quantile_dim: int) -> np.ndarray:
    """Build the [position(10) || quantile_features(quantile_dim)] vector from a state dict.

    Missing components fall back to zeros — the network sees the same shape
    regardless of whether the actor sent quantile features or whether the
    QuantileNet loader is enabled.
    """
    out = np.zeros(total_dim, dtype=np.float32)
    if not isinstance(state, dict):
        return out
    p = state.get('position')
    if p is not None:
        arr = np.asarray(p, dtype=np.float32)
        n = min(len(arr), POSITION_BASE_DIM)
        out[:n] = arr[:n]
    if quantile_dim > 0:
        q = state.get('quantile_features')
        if q is not None:
            arr = np.asarray(q, dtype=np.float32)
            n = min(len(arr), quantile_dim)
            out[POSITION_BASE_DIM:POSITION_BASE_DIM + n] = arr[:n]
    return out


class ReplayBuffer:
    """
    Pre-allocated replay buffer with pinned memory.
    Instead of a list of Python objects, the buffer is one large memory block per field.
    Pinned memory enables faster asynchronous CPU→GPU transfer via DMA.
    """

    def __init__(self, config):
        self.capacity = config['training']['buffer_capacity']
        self.num_features = config['features']['num_features']
        self.num_actions = config['model']['num_actions']
        self.device = config['learner']['device']
        # Pinned memory — cannot be swapped to disk,
        # enables faster CPU→GPU transfer via DMA (GPU-only)
        self.use_pin = torch.cuda.is_available()
        pin = (lambda t: t.pin_memory()) if self.use_pin else (lambda t: t)

        tf_cfg = config['timeframes']
        self.timeframe_keys = [k for k in sorted(tf_cfg.keys()) if tf_cfg[k] > 0]
        self.timeframe_sizes = {k: tf_cfg[k] for k in self.timeframe_keys}

        self.states = {}
        self.next_states = {}
        for key in self.timeframe_keys:
            seq_len = self.timeframe_sizes[key]
            self.states[key] = pin(torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32))
            self.next_states[key] = pin(torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32))

        self.actions = pin(torch.zeros(self.capacity, dtype=torch.long))
        self.rewards = pin(torch.zeros(self.capacity, dtype=torch.float32))
        self.dones = pin(torch.zeros(self.capacity, dtype=torch.float32))
        self.action_masks = pin(torch.zeros(self.capacity, self.num_actions, dtype=torch.float32))

        # pos_features = base position vector (10) || prerolled quantile features (0 or N).
        # Total dim derived from config so the network and buffer stay in sync.
        self.pos_dim = extended_position_dim(config)
        self.quantile_dim = self.pos_dim - POSITION_BASE_DIM
        self.pos_features      = pin(torch.zeros(self.capacity, self.pos_dim, dtype=torch.float32))
        self.next_pos_features = pin(torch.zeros(self.capacity, self.pos_dim, dtype=torch.float32))

        self.position = 0
        self.size = 0

    def _to_seq_tensor(self, state, key, tf_name, seq_len):
        """Extracts the sequence for a timeframe and normalizes shape to [seq_len, num_features]."""
        if state is None:
            return torch.zeros(seq_len, self.num_features, dtype=torch.float32)
        data = state.get(key) if key in state else state.get(tf_name)
        if not data:
            return torch.zeros(seq_len, self.num_features, dtype=torch.float32)
        t = torch.tensor(data, dtype=torch.float32)
        if t.shape == torch.Size([seq_len, self.num_features]):
            return t
        if t.ndim == 2 and t.shape[1] == self.num_features:
            out = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
            n = min(t.shape[0], seq_len)
            out[:n] = t[:n]
            return out
        return torch.zeros(seq_len, self.num_features, dtype=torch.float32)

    def add(self, state, action, reward, next_state, done, action_mask=None):
        idx = self.position

        for key in self.timeframe_keys:
            # Keys may arrive as '1m' or 'candles_1m'
            tf_name = key.replace('candles_', '')
            seq_len = self.timeframe_sizes[key]
            self.states[key][idx] = self._to_seq_tensor(state, key, tf_name, seq_len)
            self.next_states[key][idx] = self._to_seq_tensor(next_state, key, tf_name, seq_len)

        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32)
            if mask_tensor.shape[0] == self.num_actions:
                self.action_masks[idx] = mask_tensor
            else:
                # Invalid mask size — reset to all-ones
                self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)
        else:
            self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)

        self.pos_features[idx] = torch.from_numpy(
            _extract_pos_vector(state, self.pos_dim, self.quantile_dim))
        self.next_pos_features[idx] = torch.from_numpy(
            _extract_pos_vector(next_state, self.pos_dim, self.quantile_dim))

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch_add(self, experiences):
        """Batch insert — one numpy/tensor pass instead of N individual add() calls."""
        n = len(experiences)
        if n == 0:
            return
        positions = (self.position + np.arange(n, dtype=np.int64)) % self.capacity

        for key in self.timeframe_keys:
            tf_name = key.replace('candles_', '')
            seq_len = self.timeframe_sizes[key]
            s_batch = np.zeros((n, seq_len, self.num_features), dtype=np.float32)
            ns_batch = np.zeros((n, seq_len, self.num_features), dtype=np.float32)
            for i, (state, _, _, next_state, _, _) in enumerate(experiences):
                sd = state.get(key) or state.get(tf_name)
                if sd:
                    arr = np.asarray(sd, dtype=np.float32)
                    al = min(arr.shape[0], seq_len)
                    s_batch[i, :al] = arr[:al]
                if next_state is not None:
                    nd = next_state.get(key) or next_state.get(tf_name)
                    if nd:
                        arr = np.asarray(nd, dtype=np.float32)
                        al = min(arr.shape[0], seq_len)
                        ns_batch[i, :al] = arr[:al]
            self.states[key][positions] = torch.from_numpy(s_batch)
            self.next_states[key][positions] = torch.from_numpy(ns_batch)

        self.actions[positions] = torch.from_numpy(
            np.array([e[1] for e in experiences], dtype=np.int64))
        self.rewards[positions] = torch.from_numpy(
            np.array([e[2] for e in experiences], dtype=np.float32))
        self.dones[positions] = torch.from_numpy(
            np.array([float(e[4]) for e in experiences], dtype=np.float32))

        masks = np.ones((n, self.num_actions), dtype=np.float32)
        for i, e in enumerate(experiences):
            am = e[5]
            if am is not None:
                arr = np.asarray(am, dtype=np.float32)
                if len(arr) == self.num_actions:
                    masks[i] = arr
        self.action_masks[positions] = torch.from_numpy(masks)

        pf = np.zeros((n, self.pos_dim), dtype=np.float32)
        npf = np.zeros((n, self.pos_dim), dtype=np.float32)
        for i, (state, _, _, next_state, _, _) in enumerate(experiences):
            pf[i] = _extract_pos_vector(state, self.pos_dim, self.quantile_dim)
            npf[i] = _extract_pos_vector(next_state, self.pos_dim, self.quantile_dim)
        self.pos_features[positions] = torch.from_numpy(pf)
        self.next_pos_features[positions] = torch.from_numpy(npf)

        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states_batch = {}
        next_states_batch = {}
        for key in self.timeframe_keys:
            # non_blocking=True — asynchronous transfer, CPU can keep working
            states_batch[key] = self.states[key][indices].to(self.device, non_blocking=True)
            next_states_batch[key] = self.next_states[key][indices].to(self.device, non_blocking=True)

        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)
        action_masks = self.action_masks[indices].to(self.device, non_blocking=True)
        pos_features      = self.pos_features[indices].to(self.device, non_blocking=True)
        next_pos_features = self.next_pos_features[indices].to(self.device, non_blocking=True)

        return states_batch, actions, rewards, next_states_batch, dones, action_masks, pos_features, next_pos_features, indices

    def __len__(self):
        return self.size

    def is_ready(self, min_size):
        return self.size >= min_size

    def get_state(self):
        """Returns a serializable buffer state (only the filled portion)."""
        n = self.size
        return {
            'size': self.size,
            'position': self.position,
            'states': {k: self.states[k][:n].cpu().clone() for k in self.timeframe_keys},
            'next_states': {k: self.next_states[k][:n].cpu().clone() for k in self.timeframe_keys},
            'actions': self.actions[:n].cpu().clone(),
            'rewards': self.rewards[:n].cpu().clone(),
            'dones': self.dones[:n].cpu().clone(),
            'action_masks': self.action_masks[:n].cpu().clone(),
            'pos_features': self.pos_features[:n].cpu().clone(),
            'next_pos_features': self.next_pos_features[:n].cpu().clone(),
        }

    def get_reward_stats(self, recent_n: int = 10_000) -> dict:
        """Reward distribution in the buffer — globally and over the last recent_n entries."""
        n = self.size
        if n == 0:
            return {}

        rewards = self.rewards[:n].cpu().float()
        pos   = (rewards > 0).sum().item()
        neg   = (rewards < 0).sum().item()
        total = n

        stats = {
            'total':          total,
            'positive':       int(pos),
            'negative':       int(neg),
            'zero':           int(total - pos - neg),
            'positive_ratio': pos / total,
            'mean':           rewards.mean().item(),
            'std':            rewards.std().item() if total > 1 else 0.0,
        }

        # Last recent_n entries (end of the circular buffer)
        rn = min(recent_n, n)
        if self.size < self.capacity:
            # Buffer not full — most recent entries end at position-1
            start = max(0, self.position - rn)
            recent = self.rewards[start:self.position].cpu().float()
        else:
            # Buffer full — position points to the oldest entry
            end = self.position        # oldest; last rn entries are end-rn … end (with wrap)
            if end >= rn:
                recent = self.rewards[end - rn:end].cpu().float()
            else:
                recent = torch.cat([
                    self.rewards[self.capacity - (rn - end):self.capacity],
                    self.rewards[:end],
                ]).float()

        r_pos = (recent > 0).sum().item()
        r_neg = (recent < 0).sum().item()
        r_n   = len(recent)
        stats['recent'] = {
            'n':              r_n,
            'positive':       int(r_pos),
            'negative':       int(r_neg),
            'positive_ratio': r_pos / r_n if r_n > 0 else 0.0,
            'mean':           recent.mean().item() if r_n > 0 else 0.0,
        }
        return stats

    def load_state(self, state):
        """Restores the buffer from a saved state."""
        n = min(state['size'], self.capacity)
        self.size = n
        self.position = state['position'] % self.capacity
        for k in self.timeframe_keys:
            if k in state.get('states', {}):
                self.states[k][:n] = state['states'][k][:n]
                self.next_states[k][:n] = state['next_states'][k][:n]
        self.actions[:n] = state['actions'][:n]
        self.rewards[:n] = state['rewards'][:n]
        self.dones[:n] = state['dones'][:n]
        self.action_masks[:n] = state['action_masks'][:n]
        # Backward-compat: old checkpoints stored 10-D pos_features. New
        # buffer is wider (10 + quantile_dim) — pad with zeros and warn.
        def _restore_pos(target: torch.Tensor, saved: torch.Tensor) -> None:
            sd = saved.shape[1] if saved.ndim == 2 else 0
            td = target.shape[1]
            cols = min(sd, td)
            target[:n, :cols] = saved[:n, :cols]
            if sd < td:
                target[:n, sd:] = 0.0

        if 'pos_features' in state:
            _restore_pos(self.pos_features, state['pos_features'])
        if 'next_pos_features' in state:
            _restore_pos(self.next_pos_features, state['next_pos_features'])
