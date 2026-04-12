"""
AlertSystem — alerty przez Telegram przy krytycznych zdarzeniach treningowych.

Alerty:
  NAN_DETECTED       — NaN w loss lub Q-values (natychmiast)
  GRADIENT_EXPLODE   — grad_norm > próg
  ACTION_COLLAPSE    — jedna akcja dominuje > próg
  LOSS_PLATEAU       — brak poprawy loss przez N kroków
  ADVANTAGE_DEAD     — advantage_std < 0.001 przez > 500 kroków
  TRAINING_STARTED   — pierwsza notyfikacja po starcie
"""

import logging
import math
import threading
import time

logger = logging.getLogger('learner')


class AlertSystem:
    def __init__(self, config: dict):
        alerts_cfg = config.get('alerts', {})
        self.enabled = alerts_cfg.get('enabled', False)
        self._token = alerts_cfg.get('telegram_token', '')
        self._chat_id = alerts_cfg.get('telegram_chat_id', '')
        self._cooldown = alerts_cfg.get('cooldown_sec', 300)
        self._grad_threshold = alerts_cfg.get('grad_explode_threshold', 5.0)
        self._collapse_threshold = alerts_cfg.get('action_collapse_threshold', 0.95)
        self._plateau_steps = alerts_cfg.get('loss_plateau_steps', 2000)

        # Stan wewnętrzny
        self._last_sent: dict[str, float] = {}   # typ alertu → timestamp ostatniego wysłania
        self._loss_history: list[float] = []
        self._advantage_dead_count: int = 0
        self._training_started_sent: bool = False

        if self.enabled and (not self._token or not self._chat_id):
            logger.warning('[Alerts] Telegram token lub chat_id nie skonfigurowane — alerty wyłączone')
            self.enabled = False

    # ── publiczne API ─────────────────────────────────────────────────────

    def notify_training_started(self, step: int, buffer_size: int) -> None:
        if self._training_started_sent:
            return
        self._training_started_sent = True
        self._send_async('TRAINING_STARTED',
                         f'🟢 Trening rozpoczęty\nKrok: {step}\nBufor: {buffer_size:,}')

    def check_and_alert(self, metrics: dict, step: int) -> None:
        """Sprawdza metryki i wysyła alerty. Nieblokujące — wszystko w wątkach tła."""
        if not self.enabled:
            return

        loss = metrics.get('loss')
        q_mean = metrics.get('q_mean')
        grad_norm = metrics.get('grad_norm')
        action_ratios = metrics.get('action_ratios', [])
        advantage_std = metrics.get('advantage_std')

        # NaN — natychmiast, bez cooldownu
        nan_fields = []
        if loss is not None and isinstance(loss, float) and math.isnan(loss):
            nan_fields.append(f'loss={loss}')
        if q_mean is not None and isinstance(q_mean, float) and math.isnan(q_mean):
            nan_fields.append(f'q_mean={q_mean}')
        if nan_fields:
            self._send_async('NAN_DETECTED',
                             f'🔴 NaN wykryty!\nKrok: {step}\nPola: {", ".join(nan_fields)}',
                             ignore_cooldown=True)

        # Eksplodujący gradient
        if grad_norm is not None and grad_norm > self._grad_threshold:
            self._send_async('GRADIENT_EXPLODE',
                             f'⚠️ Eksplodujący gradient\nKrok: {step}\ngrad_norm={grad_norm:.3f} (próg: {self._grad_threshold})')

        # Kolaps akcji
        if action_ratios:
            max_ratio = max(action_ratios)
            max_idx = action_ratios.index(max_ratio)
            action_names = ['LONG', 'SHORT', 'HOLD', 'CLOSE']
            if max_ratio > self._collapse_threshold:
                name = action_names[max_idx] if max_idx < len(action_names) else str(max_idx)
                self._send_async('ACTION_COLLAPSE',
                                 f'⚠️ Kolaps polityki\nKrok: {step}\n{name}={max_ratio:.1%} (próg: {self._collapse_threshold:.0%})')

        # Plateau loss
        if loss is not None and not (isinstance(loss, float) and math.isnan(loss)):
            self._loss_history.append(loss)
            if len(self._loss_history) > self._plateau_steps:
                self._loss_history = self._loss_history[-self._plateau_steps:]
            if len(self._loss_history) == self._plateau_steps:
                first_half = sum(self._loss_history[:self._plateau_steps // 2]) / (self._plateau_steps // 2)
                second_half = sum(self._loss_history[self._plateau_steps // 2:]) / (self._plateau_steps // 2)
                if second_half >= first_half * 0.99:   # mniej niż 1% poprawy
                    self._send_async('LOSS_PLATEAU',
                                     f'📊 Plateau loss\nKrok: {step}\n'
                                     f'Pierwsza połowa: {first_half:.5f}\n'
                                     f'Druga połowa: {second_half:.5f}')

        # Martwy advantage stream
        if advantage_std is not None:
            if advantage_std < 0.001:
                self._advantage_dead_count += 1
            else:
                self._advantage_dead_count = 0
            if self._advantage_dead_count >= 500:
                self._send_async('ADVANTAGE_DEAD',
                                 f'⚠️ Advantage stream martwy\nKrok: {step}\n'
                                 f'advantage_std={advantage_std:.5f} przez {self._advantage_dead_count} kroków')
                self._advantage_dead_count = 0   # reset żeby nie spamować

    # ── wewnętrzne ────────────────────────────────────────────────────────

    def _send_async(self, alert_type: str, message: str, ignore_cooldown: bool = False) -> None:
        now = time.time()
        last = self._last_sent.get(alert_type, 0.0)
        if not ignore_cooldown and (now - last) < self._cooldown:
            return
        self._last_sent[alert_type] = now
        t = threading.Thread(target=self._send, args=(message,), daemon=True)
        t.start()

    def _send(self, message: str) -> None:
        try:
            import urllib.request
            import urllib.parse
            url = f'https://api.telegram.org/bot{self._token}/sendMessage'
            data = urllib.parse.urlencode({
                'chat_id': self._chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }).encode()
            req = urllib.request.Request(url, data=data, method='POST')
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f'[Alerts] Telegram HTTP {resp.status}')
        except Exception as e:
            logger.warning(f'[Alerts] Telegram send failed: {e}')
