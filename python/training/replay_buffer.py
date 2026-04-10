import torch
import numpy as np
from collections import deque


class NStepBuffer:
    """
    Lekki bufor per-aktor akumulujący n kroków przed zapisem do ReplayBuffer.

    Dla każdego kroku t emituje:
      R_t^n  = r_t + γ·r_{t+1} + … + γ^{n-1}·r_{t+n-1}
      s_{t+n}  = stan n kroków naprzód (lub stan przy done)
      done_n   = True jeśli którykolwiek krok w oknie był done
    """

    def __init__(self, n_step: int, gamma: float):
        self.n = n_step
        self.gamma = gamma
        self._deque: deque = deque()

    def add(self, state, action, reward, next_state, done, action_mask):
        """
        Dodaje przejście. Zwraca listę gotowych do zapisu transakcji (może być pusta).
        """
        self._deque.append((state, action, reward, next_state, done, action_mask))
        results = []

        if len(self._deque) >= self.n:
            results.append(self._build())
            self._deque.popleft()

        if done:
            while self._deque:
                results.append(self._build())
                self._deque.popleft()

        return results

    def _build(self):
        """Buduje n-step transition z głowy deque."""
        state_0, action_0, _, _, _, mask_0 = self._deque[0]
        n_reward = 0.0
        last_next_state = self._deque[-1][3]
        done_flag = False

        for i, (_, _, r, ns, d, _) in enumerate(self._deque):
            n_reward += (self.gamma ** i) * r
            if d:
                done_flag = True
                last_next_state = ns
                break

        return (state_0, action_0, n_reward, last_next_state, done_flag, mask_0)

    def flush(self):
        """Wymusza emisję wszystkich pozostałych przejść (np. przy resecie aktora)."""
        results = []
        while self._deque:
            results.append(self._build())
            self._deque.popleft()
        return results


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
        self.actions = a.pin_memory() if self.use_pin else a
        self.rewards = r.pin_memory() if self.use_pin else r
        self.dones = d.pin_memory() if self.use_pin else d
        self.action_masks = m.pin_memory() if self.use_pin else m

        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, action_mask=None):
        idx = self.position

        for key in self.timeframe_keys:
            # Obsługuj oba formaty kluczy:
            # 1. Bez przedrostka: '1m', '15m' (z Node.js)
            # 2. Z przedrostkiem: 'candles_1m', 'candles_15m'
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
                # Pusty lub None state - użyj zer
                self.states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
            else:
                state_tensor = torch.tensor(state_data, dtype=torch.float32)
                # Sprawdź wymiary
                if state_tensor.shape != torch.Size([seq_len, self.num_features]):
                    # Nieprawidłowe wymiary - użyj zer lub dostosuj
                    if state_tensor.ndim == 2 and state_tensor.shape[1] == self.num_features:
                        # Pierwszy wymiar się nie zgadza (paddowanie?)
                        actual_len = min(state_tensor.shape[0], seq_len)
                        zeros = torch.zeros(seq_len, self.num_features, dtype=torch.float32)
                        zeros[:actual_len] = state_tensor[:actual_len]
                        self.states[key][idx] = zeros
                    else:
                        # Całkowicie nieprawidłowe wymiary
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
                # next_state=None - zapisz zera
                self.next_states[key][idx] = torch.zeros(seq_len, self.num_features, dtype=torch.float32)

        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32)
            if mask_tensor.shape[0] == self.num_actions:
                self.action_masks[idx] = mask_tensor
            else:
                # Nieprawidłowy rozmiar maski - wyczyść
                self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)
        else:
            self.action_masks[idx] = torch.ones(self.num_actions, dtype=torch.float32)

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

    def get_state(self):
        """Zwraca serializowalny stan bufora (tylko wypełniona część)."""
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
        }

    def load_state(self, state):
        """Przywraca bufor z zapisanego stanu."""
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
