"""
TrainingReport — wysyła raport przez Telegram po każdym update sieci docelowej.

Konfiguracja przez [report] w config.toml:
  enabled          — włącz/wyłącz raporty
  mode             — "compact" (skrócony) lub "full" (pełny)
  every_n_updates  — co ile target network updates wysyłać (1 = każdy)
  min_interval_sec — minimalny czas między raportami w sekundach (0 = bez limitu)

Dane "od ostatniego raportu" są liczone jako delta między snapshotami —
niezależnie od tego czy every_n_updates wynosi 1 czy 10.
"""

import logging
import threading
import time

logger = logging.getLogger('learner')


def _arrow(current, previous, *, higher_is_better: bool = True, threshold: float = 0.02) -> str:
    """Strzałka trendu ↑/↓/→ względem poprzedniego snapshotu (próg 2%)."""
    if previous is None or previous == 0 or current is None:
        return '→'
    delta = (current - previous) / abs(previous)
    if delta > threshold:
        return '↑' if higher_is_better else '↓'
    if delta < -threshold:
        return '↓' if higher_is_better else '↑'
    return '→'


def _pct(value) -> str:
    return f'{value:.1%}' if value is not None else '—'


def _fmt_pnl(value) -> str:
    return f'{value:+.4%}' if value is not None else '—'


class TrainingReport:
    def __init__(self, config: dict):
        report_cfg = config.get('report', {})
        alerts_cfg = config.get('alerts', {})

        self.enabled          = report_cfg.get('enabled', False)
        self.mode             = report_cfg.get('mode', 'full')          # compact | full
        self.every_n_updates  = max(1, report_cfg.get('every_n_updates', 1))
        self.min_interval_sec = report_cfg.get('min_interval_sec', 0)

        # Token/chat_id z [report], fallback na [alerts]
        self._token   = report_cfg.get('telegram_token',   '') or alerts_cfg.get('telegram_token',   '')
        self._chat_id = report_cfg.get('telegram_chat_id', '') or alerts_cfg.get('telegram_chat_id', '')

        if self.enabled and (not self._token or not self._chat_id):
            logger.warning('[Report] Brak telegram_token lub chat_id — raporty wyłączone')
            self.enabled = False

        self._update_count:   int   = 0
        self._last_send_time: float = 0.0
        # Snapshot kumulatywnych wartości z momentu ostatniego raportu
        self._prev_snapshot: dict | None = None

    # ── publiczne API ─────────────────────────────────────────────────────────

    def maybe_send(self, step: int, trainer) -> None:
        """Wywołać po każdym update target network. Nieblokujące."""
        if not self.enabled:
            return

        self._update_count += 1

        if self._update_count % self.every_n_updates != 0:
            return

        now = time.time()
        if self.min_interval_sec > 0 and (now - self._last_send_time) < self.min_interval_sec:
            return

        self._last_send_time = now

        data    = self._collect(step, trainer)
        message = self._format_compact(data) if self.mode == 'compact' else self._format_full(data)

        # Zapisz snapshot PRZED wysłaniem — delta dla następnego raportu
        self._prev_snapshot = self._make_snapshot(trainer)

        threading.Thread(target=self._send, args=(message,), daemon=True).start()

    # ── snapshot / delta ──────────────────────────────────────────────────────

    def _make_snapshot(self, trainer) -> dict:
        """Zapisuje kumulatywne wartości aktorów — do obliczenia delt przy następnym raporcie."""
        snap = {'actors': {}}
        for symbol, d in trainer._actor_metrics.items():
            snap['actors'][symbol] = {
                'episode_count':        d['episode_count'],
                'episode_pnl_sum':      d['episode_pnl_sum'],
                'wins':                 d['wins'],
                'losses':               d['losses'],
                'episode_length_sum':   d['episode_length_sum'],
                'episode_length_count': d['episode_length_count'],
                'episode_trade_wins':   d['episode_trade_wins'],
                'episode_trade_losses': d['episode_trade_losses'],
                'transactions':         d['transactions'],
            }
        snap['loss_avg'] = self._current_loss_avg(trainer)
        return snap

    @staticmethod
    def _current_loss_avg(trainer) -> float | None:
        acc = trainer._metrics_accumulator
        n   = acc['steps_accumulated']
        return acc['loss_sum'] / n if n > 0 else trainer.last_loss

    # ── zbieranie danych ──────────────────────────────────────────────────────

    def _collect(self, step: int, trainer) -> dict:
        prev = self._prev_snapshot or {}

        loss_avg = self._current_loss_avg(trainer)
        buf_pct  = len(trainer.buffer) / trainer.config['training']['buffer_capacity'] * 100

        # Live metryki z akumulatora — aktualizowane co krok, nie co godzinę
        acc = trainer._metrics_accumulator
        n   = acc['steps_accumulated']
        live_q_spread = acc['q_spread_sum'] / n if n > 0 else None
        live_adv_std  = getattr(trainer, '_last_advantage_std', None)

        data = {
            'step':            step,
            'update_number':   self._update_count,
            'epsilon':         trainer.epsilon,
            'loss_avg':        loss_avg,
            'loss_arrow':      _arrow(loss_avg, prev.get('loss_avg'), higher_is_better=False),
            'learning_rate':   trainer.optimizer.param_groups[0]['lr'],
            'buffer_size':     len(trainer.buffer),
            'buffer_capacity': trainer.config['training']['buffer_capacity'],
            'buffer_pct':      buf_pct,
            'live_q_spread':   live_q_spread,
            'live_adv_std':    live_adv_std,
        }

        if trainer.use_per and hasattr(trainer.buffer, 'get_beta'):
            data['per_beta'] = trainer.buffer.get_beta(step)

        # ── Buffer reward stats ───────────────────────────────────────────────
        if hasattr(trainer.buffer, 'get_reward_stats'):
            data['reward_stats'] = trainer.buffer.get_reward_stats(recent_n=10_000)

        # ── Metryki aktorów z deltami ─────────────────────────────────────────
        prev_actors = prev.get('actors', {})
        actors = {}
        for symbol, d in trainer._actor_metrics.items():
            prev_d = prev_actors.get(symbol, {})

            # Delty od ostatniego raportu
            ep_delta    = d['episode_count']        - prev_d.get('episode_count',        0)
            pnl_delta   = d['episode_pnl_sum']      - prev_d.get('episode_pnl_sum',      0.0)
            wins_delta  = d['wins']                 - prev_d.get('wins',                 0)
            losses_delta= d['losses']               - prev_d.get('losses',               0)
            len_sum_d   = d['episode_length_sum']   - prev_d.get('episode_length_sum',   0)
            len_cnt_d   = d['episode_length_count'] - prev_d.get('episode_length_count', 0)
            tw_delta    = d['episode_trade_wins']   - prev_d.get('episode_trade_wins',   0)
            tl_delta    = d['episode_trade_losses'] - prev_d.get('episode_trade_losses', 0)
            tr_delta    = d['transactions']         - prev_d.get('transactions',         0)

            # Łączne (od początku)
            ep_total    = d['episode_count']
            tot_trades  = d['episode_trade_wins'] + d['episode_trade_losses']
            tot_wr      = d['episode_trade_wins'] / tot_trades if tot_trades > 0 else None
            tot_pnl_avg = d['episode_pnl_sum'] / ep_total if ep_total > 0 else None
            tot_len_avg = d['episode_length_sum'] / d['episode_length_count'] \
                          if d['episode_length_count'] > 0 else None

            # Od ostatniego raportu
            win_ep_delta = ep_delta > 0  # był chociaż 1 epizod
            pnl_avg_w    = pnl_delta / ep_delta if ep_delta > 0 else None
            tw_total_w   = tw_delta + tl_delta
            wr_w         = tw_delta / tw_total_w if tw_total_w > 0 else None
            len_avg_w    = len_sum_d / len_cnt_d if len_cnt_d > 0 else None

            actors[symbol] = {
                # W oknie (od ostatniego raportu)
                'window': {
                    'episodes':    ep_delta,
                    'pnl_avg':     pnl_avg_w,
                    'win_rate':    wr_w,
                    'avg_length':  len_avg_w,
                    'transactions':tr_delta,
                    'trade_wins':  tw_delta,
                    'trade_losses':tl_delta,
                },
                # Sumaryczne (od początku)
                'total': {
                    'episodes':    ep_total,
                    'pnl_avg':     tot_pnl_avg,
                    'win_rate':    tot_wr,
                    'avg_length':  tot_len_avg,
                    'transactions':d['transactions'],
                    'max_consec_losses': d['max_consecutive_losses'],
                },
                'pnl_arrow': _arrow(pnl_avg_w, None),  # brak sensownej delty dla okna
            }

        data['actors'] = actors
        data['health'] = getattr(trainer, '_last_health', None)
        return data

    # ── formatowanie ──────────────────────────────────────────────────────────

    def _format_compact(self, data: dict) -> str:
        lines = [
            f'📊 Update #{data["update_number"]}',
            f'Krok: {data["step"]:,}',
            '',
            f'📉 Loss: {data["loss_avg"]:.7f} {data["loss_arrow"]}',
            f'ε: {data["epsilon"]:.4f}',
        ]

        actors = data.get('actors', {})
        for sym, d in actors.items():
            w = d['window']
            lines += [
                '',
                f'💰 <b>{sym}</b> <i>({w["episodes"]} ep)</i>',
                f'  PnL avg: {_fmt_pnl(w["pnl_avg"])}',
                f'  WR:      {_pct(w["win_rate"])}',
                f'  Dł. ep:  {w["avg_length"]:.0f} kroków' if w['avg_length'] else '  Dł. ep:  —',
            ]

        return '\n'.join(lines)

    def _format_full(self, data: dict) -> str:
        lines = [
            f'📊 Update #{data["update_number"]}',
            f'Krok: {data["step"]:,}',
            '',
            '📉 <b>Trening</b>',
            f'  Loss avg:  {data["loss_avg"]:.7f} {data["loss_arrow"]}',
            f'  Epsilon:   {data["epsilon"]:.4f}',
            f'  LR:        {data["learning_rate"]:.2e}',
        ]

        # ── Aktorzy ──────────────────────────────────────────────────────────
        actors = data.get('actors', {})
        if actors:
            lines.append('')
            lines.append('💰 <b>Aktorzy</b>')
            for sym, d in actors.items():
                w = d['window']
                t = d['total']

                lines += [
                    f'  <b>{sym}</b>',
                    f'  ↳ <i>Okno</i>',
                    f'    Epizody:    {w["episodes"]}',
                    f'    Transakcje: {w["transactions"]}',
                    f'    PnL avg:    {_fmt_pnl(w["pnl_avg"])}',
                    f'    WR:         {_pct(w["win_rate"])}',
                    f'    Dł. ep:     {w["avg_length"]:.1f} kroków' if w['avg_length'] else '    Dł. ep:     —',
                    f'    Wygrane tr: {w["trade_wins"]}',
                    f'    Przeg. tr:  {w["trade_losses"]}',
                    f'  ↳ <i>Łącznie</i>',
                    f'    Epizody:    {t["episodes"]}',
                    f'    Transakcje: {t["transactions"]}',
                    f'    PnL avg:    {_fmt_pnl(t["pnl_avg"])}',
                    f'    WR:         {_pct(t["win_rate"])}',
                    f'    Dł. ep:     {t["avg_length"]:.1f} kroków' if t['avg_length'] else '    Dł. ep:     —',
                    f'    Max strat z rzędu: {t["max_consec_losses"]}',
                ]

        # ── Zdrowie sieci ─────────────────────────────────────────────────────
        lqs = data.get('live_q_spread')
        las = data.get('live_adv_std')
        lines += [
            '',
            '🧠 <b>Zdrowie sieci</b> <i>(live)</i>',
            f'  Q-spread: {lqs:.4f}' if lqs is not None else '  Q-spread: —',
            f'  Adv std:  {las:.4f}' if las is not None else '  Adv std:  —',
        ]

        health = data.get('health')
        if health:
            diversity = health.get('action_diversity', '?')
            dom_name  = health.get('dominant_action', '?')
            dom_ratio = health.get('dominant_ratio')
            ts        = health.get('ts')
            age_min   = int((time.time() - ts) / 60) if ts else None
            age_str   = f'{age_min} min temu' if age_min is not None else '?'
            lines += [
                f'  <i>Health check ({age_str})</i>',
                f'  Diversity:  {diversity}/4',
                f'  Dominant:   {dom_name} ({dom_ratio:.0%})' if dom_ratio is not None else f'  Dominant:   {dom_name}',
            ]
            hqs = health.get('q_spread_mean')
            has = health.get('advantage_std')
            if hqs is not None:
                lines.append(f'  Q-spread (HC): {hqs:.4f}')
            if has is not None:
                lines.append(f'  Adv std (HC):  {has:.4f}')

        # ── Buffer ────────────────────────────────────────────────────────────
        lines += [
            '',
            '🗄️ <b>Buffer</b>',
            f'  Rozmiar:  {data["buffer_size"]:,}',
            f'  Pojemność:{data["buffer_capacity"]:,}',
            f'  Wypełn.:  {data["buffer_pct"]:.1f}%',
        ]
        if 'per_beta' in data:
            lines.append(f'  PER beta: {data["per_beta"]:.3f}')

        rs = data.get('reward_stats', {})
        if rs:
            total = max(rs.get('total', 1), 1)
            lines += [
                '',
                '📦 <b>Rozkład nagród</b>',
                f'  Pozytywne: {_pct(rs.get("positive_ratio"))} ({rs.get("positive", 0):,})',
                f'  Negatywne: {_pct(rs.get("negative", 0) / total)} ({rs.get("negative", 0):,})',
                f'  Zerowe:    {rs.get("zero", 0):,}',
                f'  Średnia:   {rs.get("mean", 0):+.5f}',
                f'  Odch. std: {rs.get("std", 0):.5f}',
            ]
            recent = rs.get('recent', {})
            if recent:
                lines += [
                    f'  <i>Ostatnie {recent.get("n", 0):,} wpisów</i>',
                    f'    Pozytywne: {_pct(recent.get("positive_ratio"))}',
                    f'    Średnia:   {recent.get("mean", 0):+.5f}',
                ]

        return '\n'.join(lines)

    # ── wysyłanie ─────────────────────────────────────────────────────────────

    def _send(self, message: str) -> None:
        try:
            import urllib.request
            import urllib.parse
            url  = f'https://api.telegram.org/bot{self._token}/sendMessage'
            body = urllib.parse.urlencode({
                'chat_id':    self._chat_id,
                'text':       message,
                'parse_mode': 'HTML',
            }).encode()
            req = urllib.request.Request(url, data=body, method='POST')
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f'[Report] Telegram HTTP {resp.status}')
        except Exception as e:
            logger.warning(f'[Report] Telegram send failed: {e}')
