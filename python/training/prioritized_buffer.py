import copy

import numpy as np
import torch


class SumTree:
    """
    SumTree — binary tree for efficient priority-weighted sampling.
    Leaves store priorities, internal nodes store the sum of their children.
    Sample and update are O(log n) instead of O(n).

    Iterative (non-recursive) implementation + vectorized batch ops.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def _propagate(self, idx, change):
        # Iteratively walk up the tree — no recursion (faster, no stack overhead)
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        # Iterative version — no recursion
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    def _retrieve_batch(self, s_array):
        """Vectorized batch sampling — one numpy pass instead of N Python calls."""
        idx = np.zeros(len(s_array), dtype=np.int64)
        s = s_array.copy()
        tree_len = len(self.tree)
        while True:
            left = 2 * idx + 1
            terminal = left >= tree_len
            if terminal.all():
                break
            safe_left = np.minimum(left, tree_len - 1)
            left_vals = np.where(terminal, 0.0, self.tree[safe_left])
            go_right = (~terminal) & (s > left_vals)
            s[go_right] -= left_vals[go_right]
            idx = np.where(terminal, idx, np.where(go_right, left + 1, left))
        return idx

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

    def update_batch(self, data_indices, priorities):
        """Batch priority update — vectorized propagation up the tree."""
        tree_indices = (data_indices + self.capacity - 1).astype(np.int64)
        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        idx = tree_indices.copy()
        while idx.max() > 0:
            mask = idx > 0
            idx[mask] = (idx[mask] - 1) // 2
            np.add.at(self.tree, idx[mask], changes[mask])

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
    Prioritized Experience Replay with SumTree.
    Experiences with larger TD error are sampled more often.
    Importance Sampling weights correct the resulting bias.
    """

    _MAX_PRIORITY = 1e6  # hard cap — prevents Inf propagating into the tree

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
        pin = (lambda t: t.pin_memory()) if self.use_pin else (lambda t: t)

        tf_cfg = config['timeframes']
        self.timeframe_keys = [k for k in sorted(tf_cfg.keys()) if tf_cfg[k] > 0]
        self.timeframe_sizes = {k: tf_cfg[k] for k in self.timeframe_keys}

        self.states = {}
        self.next_states = {}
        for key in self.timeframe_keys:
            seq_len = self.timeframe_sizes[key]
            self.states[key]      = pin(torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32))
            self.next_states[key] = pin(torch.zeros(self.capacity, seq_len, self.num_features, dtype=torch.float32))

        self.actions           = pin(torch.zeros(self.capacity, dtype=torch.long))
        self.rewards           = pin(torch.zeros(self.capacity, dtype=torch.float32))
        self.dones             = pin(torch.zeros(self.capacity, dtype=torch.float32))
        self.action_masks      = pin(torch.zeros(self.capacity, self.num_actions, dtype=torch.float32))
        self.pos_features      = pin(torch.zeros(self.capacity, 10, dtype=torch.float32))
        self.next_pos_features = pin(torch.zeros(self.capacity, 10, dtype=torch.float32))

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, action_mask=None, td_error=None):
        idx = self.position

        for key in self.timeframe_keys:
            tf_name = key.replace('candles_', '')
            seq_len = self.timeframe_sizes[key]
            
            # Fetch state for this timeframe
            state_data = None
            if key in state:
                state_data = state[key]
            elif tf_name in state:
                state_data = state[tf_name]
            
            # Validate and convert state
            if state_data is None or len(state_data) == 0:
                self.states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
            else:
                state_tensor = torch.tensor(state_data, dtype=torch.float32)
                if state_tensor.shape != torch.Size([seq_len, self.num_features]):
                    if state_tensor.ndim == 2 and state_tensor.shape[1] == self.num_features:
                        actual_len = min(state_tensor.shape[0], seq_len)
                        zeros = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                        zeros[:actual_len] = state_tensor[:actual_len]
                        self.states[key][idx] = zeros
                    else:
                        self.states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                else:
                    self.states[key][idx] = state_tensor
            
            # Fetch next_state for this timeframe
            if next_state is not None:
                next_state_data = None
                if key in next_state:
                    next_state_data = next_state[key]
                elif tf_name in next_state:
                    next_state_data = next_state[tf_name]
                
                if next_state_data is None or len(next_state_data) == 0:
                    self.next_states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                else:
                    next_tensor = torch.tensor(next_state_data, dtype=torch.float32)
                    if next_tensor.shape != torch.Size([seq_len, self.num_features]):
                        if next_tensor.ndim == 2 and next_tensor.shape[1] == self.num_features:
                            actual_len = min(next_tensor.shape[0], seq_len)
                            zeros = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                            zeros[:actual_len] = next_tensor[:actual_len]
                            self.next_states[key][idx] = zeros
                        else:
                            self.next_states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                    else:
                        self.next_states[key][idx] = next_tensor
            else:
                self.next_states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)

        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32)
            if mask_tensor.shape[0] == self.num_actions:
                self.action_masks[idx] = mask_tensor
            else:
                self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)
        else:
            self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)

        pf = state.get('position') if isinstance(state, dict) else None
        if pf is not None:
            arr = torch.zeros(10, dtype=torch.float32)
            src = torch.tensor(pf[:10], dtype=torch.float32)
            arr[:src.shape[0]] = src
            self.pos_features[idx] = arr
        else:
            self.pos_features[idx] = torch.zeros(10, dtype=torch.float32)

        npf = next_state.get('position') if isinstance(next_state, dict) else None
        if npf is not None:
            arr = torch.zeros(10, dtype=torch.float32)
            src = torch.tensor(npf[:10], dtype=torch.float32)
            arr[:src.shape[0]] = src
            self.next_pos_features[idx] = arr
        else:
            self.next_pos_features[idx] = torch.zeros(10, dtype=torch.float32)

        if td_error is not None:
            td_val = abs(td_error) if np.isfinite(td_error) else self.max_priority
            priority = min((td_val + self.per_epsilon) ** self.alpha, self._MAX_PRIORITY)
        else:
            priority = min(self.max_priority ** self.alpha, self._MAX_PRIORITY)

        self.tree.add(priority)
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

        pf = np.zeros((n, 10), dtype=np.float32)
        npf = np.zeros((n, 10), dtype=np.float32)
        for i, (state, _, _, next_state, _, _) in enumerate(experiences):
            p = state.get('position') if isinstance(state, dict) else None
            if p is not None:
                arr = np.asarray(p[:10], dtype=np.float32)
                pf[i, :arr.shape[0]] = arr
            next_pos = next_state.get('position') if isinstance(next_state, dict) else None
            if next_pos is not None:
                arr = np.asarray(next_pos[:10], dtype=np.float32)
                npf[i, :arr.shape[0]] = arr
        self.pos_features[positions] = torch.from_numpy(pf)
        self.next_pos_features[positions] = torch.from_numpy(npf)

        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

        priority = min(self.max_priority ** self.alpha, self._MAX_PRIORITY)
        priorities = np.full(n, priority, dtype=np.float64)
        self.tree.update_batch(positions, priorities)
        self.tree.data_pointer = self.position

    def _sanitize_tree(self):
        """Reset tree to uniform priorities if it contains invalid values."""
        total = self.tree.total()
        if not np.isfinite(total) or total <= 0:
            uniform_priority = min(self.max_priority ** self.alpha, self._MAX_PRIORITY)
            if not np.isfinite(uniform_priority) or uniform_priority <= 0:
                uniform_priority = 1.0
            self.tree.tree[:] = 0.0
            self.max_priority = uniform_priority
            if self.size > 0:
                # Batch update: ~500K Python-level update() calls caused a noticeable hang
                # with a full buffer. update_batch does it in a single vectorized pass.
                indices = np.arange(self.size, dtype=np.int64)
                priorities = np.full(self.size, uniform_priority, dtype=np.float64)
                self.tree.update_batch(indices, priorities)

    def sample(self, batch_size):
        self._sanitize_tree()
        total = self.tree.total()
        segment = total / batch_size

        # Vectorized sampling: a single numpy batch instead of a Python loop
        a = np.arange(batch_size, dtype=np.float64) * segment
        b = a + segment
        s_array = np.random.uniform(a, b)
        tree_indices = self.tree._retrieve_batch(s_array)
        indices = (tree_indices - self.tree.capacity + 1).astype(np.int64)
        priorities = self.tree.tree[tree_indices]

        sampling_probs = priorities / total
        sampling_probs = np.clip(sampling_probs, 1e-10, None)

        is_weights = (self.size * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights.astype(np.float32)).to(self.device, non_blocking=True)

        states_batch = {}
        next_states_batch = {}
        for key in self.timeframe_keys:
            states_batch[key] = self.states[key][indices].to(self.device, non_blocking=True)
            next_states_batch[key] = self.next_states[key][indices].to(self.device, non_blocking=True)

        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)
        action_masks = self.action_masks[indices].to(self.device, non_blocking=True)
        pos_features      = self.pos_features[indices].to(self.device, non_blocking=True)
        next_pos_features = self.next_pos_features[indices].to(self.device, non_blocking=True)

        return states_batch, actions, rewards, next_states_batch, dones, action_masks, pos_features, next_pos_features, indices, is_weights

    def update_priorities(self, indices, td_errors):
        td_vals = np.abs(td_errors).astype(np.float64)
        td_vals = np.where(np.isfinite(td_vals), td_vals, self.max_priority)
        priorities = np.minimum((td_vals + self.per_epsilon) ** self.alpha, self._MAX_PRIORITY)
        self.tree.update_batch(indices.astype(np.int64), priorities)
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def update_beta(self, fraction):
        self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * fraction)

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
            'max_priority': self.max_priority,
            'beta': self.beta,
            'tree': self.tree.tree.copy(),
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
        """Reward distribution — identical logic to ReplayBuffer."""
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
        rn = min(recent_n, n)
        if self.size < self.capacity:
            start  = max(0, self.position - rn)
            recent = self.rewards[start:self.position].cpu().float()
        else:
            end = self.position
            if end >= rn:
                recent = self.rewards[end - rn:end].cpu().float()
            else:
                recent = torch.cat([
                    self.rewards[self.capacity - (rn - end):self.capacity],
                    self.rewards[:end],
                ]).float()
        r_pos = (recent > 0).sum().item()
        r_n   = len(recent)
        stats['recent'] = {
            'n':              r_n,
            'positive':       int(r_pos),
            'negative':       int((recent < 0).sum().item()),
            'positive_ratio': r_pos / r_n if r_n > 0 else 0.0,
            'mean':           recent.mean().item() if r_n > 0 else 0.0,
        }
        return stats

    def load_state(self, state):
        """Restores the buffer from a saved state."""
        n = min(state['size'], self.capacity)
        self.size = n
        self.position = state['position'] % self.capacity
        self.max_priority = state.get('max_priority', 1.0)
        self.beta = state.get('beta', self.beta_start)
        if 'tree' in state and len(state['tree']) == len(self.tree.tree):
            self.tree.tree[:] = state['tree']
            self.tree.data_pointer = self.position
        self._sanitize_tree()  # fix Inf/NaN from previous runs
        for k in self.timeframe_keys:
            if k in state.get('states', {}):
                self.states[k][:n] = state['states'][k][:n]
                self.next_states[k][:n] = state['next_states'][k][:n]
        self.actions[:n] = state['actions'][:n]
        self.rewards[:n] = state['rewards'][:n]
        self.dones[:n] = state['dones'][:n]
        self.action_masks[:n] = state['action_masks'][:n]
        if 'pos_features' in state:
            self.pos_features[:n] = state['pos_features'][:n]
        if 'next_pos_features' in state:
            self.next_pos_features[:n] = state['next_pos_features'][:n]


class DualPrioritizedBuffer:
    """
    Dual Buffer with a guaranteed proportion of profitable samples per batch.

    Composed of two buffers:
    - main:     full-capacity PrioritizedReplayBuffer — all experiences, PER
    - positive: circular buffer (1/4 capacity) — only reward > 0, uniform sampling

    Each sample() guarantees a minimum fraction of samples from the positive buffer
    (positive_ratio in [per]). When the positive buffer is empty or too small,
    we fall back to sampling only from main (no error).

    update_priorities() updates the main buffer only — positive-buffer indices are
    encoded as idx + capacity so they can be identified and skipped.
    """

    def __init__(self, config):
        per_cfg = config['per']
        self.positive_ratio = per_cfg.get('positive_ratio', 0.4)
        self.long_short_balance = per_cfg.get('long_short_balance', 0.0)
        self.capacity = config['training']['buffer_capacity']

        self.main = PrioritizedReplayBuffer(config)

        # Positive buffer: 1/4 capacity, no PER (uniform sampling)
        pos_config = copy.deepcopy(config)
        pos_config['training']['buffer_capacity'] = max(1, self.capacity // 4)
        # Import here to avoid circular imports at module load time
        from training.replay_buffer import ReplayBuffer
        self.positive = ReplayBuffer(pos_config)
        self._pos_capacity = pos_config['training']['buffer_capacity']

        # Directional PER buffers: separate PER for LONG (action=0) and SHORT (action=1).
        # Capacity = capacity/4 — each direction is roughly half of the experience stream,
        # but a smaller buffer keeps the directional PER responsive to recent samples.
        # Index encoding in update_priorities:
        #   main:     idx <  capacity
        #   positive: idx ∈ [capacity,   2*capacity)  — uniform, no priority update
        #   long:     idx ∈ [2*capacity, 3*capacity)  (idx - 2*capacity)
        #   short:    idx ∈ [3*capacity, 4*capacity)  (idx - 3*capacity)
        self.long_buf = None
        self.short_buf = None
        if self.long_short_balance > 0.0:
            dir_config = copy.deepcopy(config)
            dir_config['training']['buffer_capacity'] = max(1, self.capacity // 4)
            self.long_buf  = PrioritizedReplayBuffer(dir_config)
            self.short_buf = PrioritizedReplayBuffer(dir_config)
            self._dir_capacity = dir_config['training']['buffer_capacity']

    # ── Delegates to main ─────────────────────────────────────────────────────
    @property
    def beta(self):
        return self.main.beta

    @property
    def max_priority(self):
        return self.main.max_priority

    @property
    def size(self):
        return self.main.size

    @property
    def tree(self):
        """SumTree access for TensorBoard logging (delegates to main)."""
        return self.main.tree

    def update_beta(self, fraction):
        self.main.update_beta(fraction)

    def __len__(self):
        return len(self.main)

    def is_ready(self, min_size):
        return self.main.is_ready(min_size)

    def get_reward_stats(self, recent_n: int = 10_000) -> dict:
        return self.main.get_reward_stats(recent_n=recent_n)

    # ── Add ───────────────────────────────────────────────────────────────────
    def add(self, state, action, reward, next_state, done, action_mask=None):
        self.main.add(state, action, reward, next_state, done, action_mask)
        if reward > 0:
            self.positive.add(state, action, reward, next_state, done, action_mask)
        if self.long_buf is not None:
            if action == 0:
                self.long_buf.add(state, action, reward, next_state, done, action_mask)
            elif action == 1:
                self.short_buf.add(state, action, reward, next_state, done, action_mask)

    def batch_add(self, experiences):
        self.main.batch_add(experiences)
        pos_exps = [e for e in experiences if e[2] > 0]
        if pos_exps:
            self.positive.batch_add(pos_exps)
        if self.long_buf is not None:
            long_exps  = [e for e in experiences if e[1] == 0]
            short_exps = [e for e in experiences if e[1] == 1]
            if long_exps:
                self.long_buf.batch_add(long_exps)
            if short_exps:
                self.short_buf.batch_add(short_exps)

    # ── Sample ────────────────────────────────────────────────────────────────
    def sample(self, batch_size):
        # ── Compute sample split ───────────────────────────────────────────
        n_pos = 0
        if len(self.positive) > 0 and self.positive_ratio > 0:
            n_pos = min(int(batch_size * self.positive_ratio), len(self.positive))

        n_long = n_short = 0
        if (self.long_buf is not None
                and len(self.long_buf) > 0 and len(self.short_buf) > 0
                and self.long_short_balance > 0):
            half = int(batch_size * self.long_short_balance / 2)
            n_long  = min(half, len(self.long_buf))
            n_short = min(half, len(self.short_buf))

        n_main = batch_size - n_pos - n_long - n_short
        if n_main < 1:
            n_pos = n_long = n_short = 0
            n_main = batch_size

        # ── Sample from main (PER) ─────────────────────────────────────────
        (states_m, actions_m, rewards_m, next_states_m, dones_m,
         masks_m, pf_m, npf_m, idx_m, iw_m) = self.main.sample(n_main)

        if n_pos == 0 and n_long == 0 and n_short == 0:
            return (states_m, actions_m, rewards_m, next_states_m, dones_m,
                    masks_m, pf_m, npf_m, idx_m, iw_m)

        device = self.main.device
        parts_s, parts_ns = {k: [states_m[k]] for k in states_m}, {k: [next_states_m[k]] for k in states_m}
        parts_a   = [actions_m]
        parts_r   = [rewards_m]
        parts_d   = [dones_m]
        parts_mk  = [masks_m]
        parts_pf  = [pf_m]
        parts_npf = [npf_m]
        parts_iw  = [iw_m]
        parts_idx = [idx_m]

        # ── Positive buffer (uniform, no PER) ──────────────────────────────
        if n_pos > 0:
            (states_p, actions_p, rewards_p, next_states_p, dones_p,
             masks_p, pf_p, npf_p, idx_p) = self.positive.sample(n_pos)
            for k in parts_s: parts_s[k].append(states_p[k]); parts_ns[k].append(next_states_p[k])
            parts_a.append(actions_p); parts_r.append(rewards_p); parts_d.append(dones_p)
            parts_mk.append(masks_p); parts_pf.append(pf_p); parts_npf.append(npf_p)
            parts_iw.append(torch.ones(n_pos, device=device, dtype=torch.float32))
            parts_idx.append(idx_p + self.capacity)

        # ── Long buffer (PER) ─────────────────────────────────────────────
        if n_long > 0:
            (states_l, actions_l, rewards_l, next_states_l, dones_l,
             masks_l, pf_l, npf_l, idx_l, iw_l) = self.long_buf.sample(n_long)
            for k in parts_s: parts_s[k].append(states_l[k]); parts_ns[k].append(next_states_l[k])
            parts_a.append(actions_l); parts_r.append(rewards_l); parts_d.append(dones_l)
            parts_mk.append(masks_l); parts_pf.append(pf_l); parts_npf.append(npf_l)
            parts_iw.append(iw_l)
            parts_idx.append(idx_l + 2 * self.capacity)

        # ── Short buffer (PER) ────────────────────────────────────────────
        if n_short > 0:
            (states_sh, actions_sh, rewards_sh, next_states_sh, dones_sh,
             masks_sh, pf_sh, npf_sh, idx_sh, iw_sh) = self.short_buf.sample(n_short)
            for k in parts_s: parts_s[k].append(states_sh[k]); parts_ns[k].append(next_states_sh[k])
            parts_a.append(actions_sh); parts_r.append(rewards_sh); parts_d.append(dones_sh)
            parts_mk.append(masks_sh); parts_pf.append(pf_sh); parts_npf.append(npf_sh)
            parts_iw.append(iw_sh)
            parts_idx.append(idx_sh + 3 * self.capacity)

        # ── Merge ──────────────────────────────────────────────────────────
        states      = {k: torch.cat(parts_s[k],   dim=0) for k in parts_s}
        next_states = {k: torch.cat(parts_ns[k],  dim=0) for k in parts_ns}
        actions      = torch.cat(parts_a,   dim=0)
        rewards      = torch.cat(parts_r,   dim=0)
        dones        = torch.cat(parts_d,   dim=0)
        masks        = torch.cat(parts_mk,  dim=0)
        pf           = torch.cat(parts_pf,  dim=0)
        npf          = torch.cat(parts_npf, dim=0)
        is_weights   = torch.cat(parts_iw,  dim=0)
        is_weights   = is_weights / is_weights.max()
        indices      = np.concatenate(parts_idx)

        return (states, actions, rewards, next_states, dones,
                masks, pf, npf, indices, is_weights)

    # ── Priority updates ──────────────────────────────────────────────────────
    def update_priorities(self, indices, td_errors):
        """Updates priorities in all active PER buffers.
        Encoding: main < capacity, positive [capacity, 2*cap),
                  long [2*cap, 3*cap), short [3*cap, 4*cap)."""
        main_mask = indices < self.capacity
        if main_mask.any():
            self.main.update_priorities(indices[main_mask], td_errors[main_mask])
        if self.long_buf is not None:
            long_mask  = (indices >= 2 * self.capacity) & (indices < 3 * self.capacity)
            short_mask = (indices >= 3 * self.capacity) & (indices < 4 * self.capacity)
            if long_mask.any():
                self.long_buf.update_priorities(indices[long_mask] - 2 * self.capacity, td_errors[long_mask])
            if short_mask.any():
                self.short_buf.update_priorities(indices[short_mask] - 3 * self.capacity, td_errors[short_mask])

    # ── Checkpoint ───────────────────────────────────────────────────────────
    def get_state(self):
        state = {
            'main':     self.main.get_state(),
            'positive': self.positive.get_state(),
        }
        if self.long_buf is not None:
            state['long']  = self.long_buf.get_state()
            state['short'] = self.short_buf.get_state()
        return state

    def load_state(self, state):
        if 'main' in state:
            self.main.load_state(state['main'])
        if 'positive' in state:
            self.positive.load_state(state['positive'])
        # Backwards-compat: old checkpoint was main state directly (no nesting)
        elif 'size' in state:
            self.main.load_state(state)
        if self.long_buf is not None and 'long' in state:
            self.long_buf.load_state(state['long'])
            self.short_buf.load_state(state['short'])
