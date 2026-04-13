import copy

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
        self.timeframe_keys = [k for k in sorted(tf_cfg.keys()) if tf_cfg[k] > 0]
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
        pf  = torch.zeros(self.capacity, 4, dtype=torch.float32)
        npf = torch.zeros(self.capacity, 4, dtype=torch.float32)
        self.actions = a.pin_memory() if self.use_pin else a
        self.rewards = r.pin_memory() if self.use_pin else r
        self.dones = d.pin_memory() if self.use_pin else d
        self.action_masks = m.pin_memory() if self.use_pin else m
        self.pos_features      = pf.pin_memory()  if self.use_pin else pf
        self.next_pos_features = npf.pin_memory() if self.use_pin else npf

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, action_mask=None, td_error=None):
        idx = self.position

        for key in self.timeframe_keys:
            tf_name = key.replace('candles_', '')
            seq_len = self.timeframe_sizes[key]
            
            # Pobierz state dla tego timeframe'a
            state_data = None
            if key in state:
                state_data = state[key]
            elif tf_name in state:
                state_data = state[tf_name]
            
            # Waliduj i konwertuj state
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
            
            # Pobierz next_state dla tego timeframe'a
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
        self.pos_features[idx] = torch.tensor(pf[:4], dtype=torch.float32) if pf is not None else torch.zeros(4)

        npf = next_state.get('position') if isinstance(next_state, dict) else None
        self.next_pos_features[idx] = torch.tensor(npf[:4], dtype=torch.float32) if npf is not None else torch.zeros(4)

        if td_error is not None:
            td_val = abs(td_error) if np.isfinite(td_error) else self.max_priority
            priority = min((td_val + self.per_epsilon) ** self.alpha, self._MAX_PRIORITY)
        else:
            priority = min(self.max_priority ** self.alpha, self._MAX_PRIORITY)

        self.tree.add(priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sanitize_tree(self):
        """Reset tree to uniform priorities if it contains invalid values."""
        total = self.tree.total()
        if not np.isfinite(total) or total <= 0:
            uniform_priority = min(self.max_priority ** self.alpha, self._MAX_PRIORITY)
            if not np.isfinite(uniform_priority) or uniform_priority <= 0:
                uniform_priority = 1.0
            self.tree.tree[:] = 0.0
            self.max_priority = uniform_priority
            for i in range(self.size):
                self.tree.update(i, uniform_priority)

    def sample(self, batch_size):
        self._sanitize_tree()
        indices = []
        priorities = []
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            _, priority, data_idx = self.tree.get(s)
            indices.append(data_idx)
            priorities.append(priority)

        indices = np.array(indices)
        priorities = np.array(priorities, dtype=np.float64)

        sampling_probs = priorities / total
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
        pos_features      = self.pos_features[indices].to(self.device, non_blocking=True)
        next_pos_features = self.next_pos_features[indices].to(self.device, non_blocking=True)

        return states_batch, actions, rewards, next_states_batch, dones, action_masks, pos_features, next_pos_features, indices, is_weights

    _MAX_PRIORITY = 1e6  # hard cap — prevents Inf propagating into the tree

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            td_val = abs(td_err)
            if not np.isfinite(td_val):
                td_val = self.max_priority
            priority = min((td_val + self.per_epsilon) ** self.alpha, self._MAX_PRIORITY)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, fraction):
        self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * fraction)

    def __len__(self):
        return self.size

    def is_ready(self, min_size):
        return self.size >= min_size

    def get_state(self):
        """Zwraca serializowalny stan bufora (tylko wypełniona część)."""
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

    def load_state(self, state):
        """Przywraca bufor z zapisanego stanu."""
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
    Dual Buffer z gwarantowaną proporcją zyskownych próbek w batchu.

    Składa się z dwóch buforów:
    - main:     PrioritizedReplayBuffer pełnej pojemności — wszystkie doświadczenia, PER
    - positive: okrągły bufor (1/4 pojemności) — tylko reward > 0, próbkowanie jednostajne

    Przy każdym sample() gwarantowana jest minimalna frakcja próbek z positive buffer
    (parametr positive_ratio w [per]). Gdy positive buffer jest pusty lub za mały —
    fallback na same próbki z main (brak błędu).

    update_priorities() aktualizuje wyłącznie main buffer — indeksy z positive buffer
    są enkodowane jako idx + capacity, co pozwala je odróżnić i pominąć.
    """

    def __init__(self, config):
        per_cfg = config['per']
        self.positive_ratio = per_cfg.get('positive_ratio', 0.4)
        self.capacity = config['training']['buffer_capacity']

        self.main = PrioritizedReplayBuffer(config)

        # Pozytywny bufor: 1/4 pojemności, bez PER (uniform)
        pos_config = copy.deepcopy(config)
        pos_config['training']['buffer_capacity'] = max(1, self.capacity // 4)
        # Importujemy tutaj, żeby uniknąć cyklicznych importów na poziomie modułu
        from training.replay_buffer import ReplayBuffer
        self.positive = ReplayBuffer(pos_config)

        self._pos_capacity = pos_config['training']['buffer_capacity']

    # ── Delegaty do main ──────────────────────────────────────────────────────
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
        """Dostęp do drzewa SumTree dla logowania TensorBoard (delegacja do main)."""
        return self.main.tree

    def update_beta(self, fraction):
        self.main.update_beta(fraction)

    def __len__(self):
        return len(self.main)

    def is_ready(self, min_size):
        return self.main.is_ready(min_size)

    # ── Add ───────────────────────────────────────────────────────────────────
    def add(self, state, action, reward, next_state, done, action_mask=None):
        self.main.add(state, action, reward, next_state, done, action_mask)
        if reward > 0:
            self.positive.add(state, action, reward, next_state, done, action_mask)

    # ── Sample ────────────────────────────────────────────────────────────────
    def sample(self, batch_size):
        n_pos = 0
        if len(self.positive) > 0 and self.positive_ratio > 0:
            n_pos = min(int(batch_size * self.positive_ratio), len(self.positive))

        n_main = batch_size - n_pos
        if n_main < 1:
            # Fallback: weź wszystko z main (nie powinno się zdarzyć przy ratio < 1.0)
            n_pos = 0
            n_main = batch_size

        # Próbkuj z main (PER) — zwraca 10 elementów
        (states_m, actions_m, rewards_m, next_states_m, dones_m,
         masks_m, pf_m, npf_m, idx_m, iw_m) = self.main.sample(n_main)

        if n_pos == 0:
            return (states_m, actions_m, rewards_m, next_states_m, dones_m,
                    masks_m, pf_m, npf_m, idx_m, iw_m)

        # Próbkuj z positive (uniform) — zwraca 9 elementów (bez is_weights)
        (states_p, actions_p, rewards_p, next_states_p, dones_p,
         masks_p, pf_p, npf_p, idx_p) = self.positive.sample(n_pos)

        device = self.main.device

        # Scal słowniki stanów
        states = {k: torch.cat([states_m[k], states_p[k]], dim=0)
                  for k in states_m}
        next_states = {k: torch.cat([next_states_m[k], next_states_p[k]], dim=0)
                       for k in next_states_m}

        actions      = torch.cat([actions_m,  actions_p],  dim=0)
        rewards      = torch.cat([rewards_m,  rewards_p],  dim=0)
        dones        = torch.cat([dones_m,    dones_p],    dim=0)
        masks        = torch.cat([masks_m,    masks_p],    dim=0)
        pf           = torch.cat([pf_m,       pf_p],       dim=0)
        npf          = torch.cat([npf_m,      npf_p],      dim=0)

        # Wagi IS: main dostaje wagi PER, positive dostaje 1.0 (uniform)
        # Re-normalizacja tak żeby max = 1.0 (standard IS-weight convention)
        iw_pos = torch.ones(n_pos, device=device, dtype=torch.float32)
        is_weights = torch.cat([iw_m, iw_pos], dim=0)
        is_weights = is_weights / is_weights.max()

        # Enkoduj indeksy: main → idx_m (< capacity),
        #                  positive → idx_p + capacity (do odróżnienia w update_priorities)
        idx_p_offset = idx_p + self.capacity
        indices = np.concatenate([idx_m, idx_p_offset])

        return (states, actions, rewards, next_states, dones,
                masks, pf, npf, indices, is_weights)

    # ── Priority updates ──────────────────────────────────────────────────────
    def update_priorities(self, indices, td_errors):
        """Aktualizuje priorytety wyłącznie w main buffer.
        Indeksy z positive buffer (>= capacity) są pomijane."""
        main_mask = indices < self.capacity
        main_indices = indices[main_mask]
        main_td = td_errors[main_mask]
        if len(main_indices) > 0:
            self.main.update_priorities(main_indices, main_td)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    def get_state(self):
        return {
            'main':     self.main.get_state(),
            'positive': self.positive.get_state(),
        }

    def load_state(self, state):
        if 'main' in state:
            self.main.load_state(state['main'])
        if 'positive' in state:
            self.positive.load_state(state['positive'])
        # Backwards-compat: old checkpoint jest stanem main (bez zagnieżdżenia)
        elif 'size' in state:
            self.main.load_state(state)
