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

import numpy as np

logger = logging.getLogger('learner')


# ── funkcje statystyczne ──────────────────────────────────────────────────────

def _max_drawdown(pnl_list: list) -> float:
    """Maksymalny drawdown equity curve zbudowanej z listy PnL epizodów."""
    if len(pnl_list) < 2:
        return 0.0
    equity = np.cumsum(pnl_list)
    peak   = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def _sharpe_sortino(pnl_list: list) -> tuple:
    """
    Zwraca (sharpe, sortino). Risk-free rate = 0.
    Wymaga min. 5 epizodów — inaczej (None, None).
    """
    if len(pnl_list) < 5:
        return None, None
    arr  = np.array(pnl_list, dtype=float)
    mu   = arr.mean()
    std  = arr.std()
    sharpe  = mu / std if std > 1e-10 else None
    neg     = arr[arr < 0]
    d_std   = neg.std() if len(neg) > 1 else 0.0
    sortino = mu / d_std if d_std > 1e-10 else None
    return sharpe, sortino


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
                # Rozszerzone statystyki handlowe
                'profit_sum':           d['profit_sum'],
                'loss_sum':             d['loss_sum'],
                'win_hold_steps':       d['win_hold_steps'],
                'loss_hold_steps':      d['loss_hold_steps'],
                'mfe_sum':              d['mfe_sum'],
                'long_opens':           d['long_opens'],
                'short_opens':          d['short_opens'],
                'action_long':          d['action_long'],
                'action_short':         d['action_short'],
                'action_hold':          d['action_hold'],
                'action_close':         d['action_close'],
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

            # ── Delty od ostatniego raportu ───────────────────────────────────
            ep_delta      = d['episode_count']        - prev_d.get('episode_count',        0)
            pnl_delta     = d['episode_pnl_sum']      - prev_d.get('episode_pnl_sum',      0.0)
            len_sum_d     = d['episode_length_sum']   - prev_d.get('episode_length_sum',   0)
            len_cnt_d     = d['episode_length_count'] - prev_d.get('episode_length_count', 0)
            tw_delta      = d['episode_trade_wins']   - prev_d.get('episode_trade_wins',   0)
            tl_delta      = d['episode_trade_losses'] - prev_d.get('episode_trade_losses', 0)
            tr_delta      = d['transactions']         - prev_d.get('transactions',         0)
            profit_d      = d['profit_sum']           - prev_d.get('profit_sum',           0.0)
            loss_d        = d['loss_sum']             - prev_d.get('loss_sum',             0.0)
            whs_d         = d['win_hold_steps']       - prev_d.get('win_hold_steps',       0)
            lhs_d         = d['loss_hold_steps']      - prev_d.get('loss_hold_steps',      0)
            mfe_d         = d['mfe_sum']              - prev_d.get('mfe_sum',              0.0)
            long_d        = d['long_opens']           - prev_d.get('long_opens',           0)
            short_d       = d['short_opens']          - prev_d.get('short_opens',          0)
            act_long_d    = d['action_long']          - prev_d.get('action_long',          0)
            act_short_d   = d['action_short']         - prev_d.get('action_short',         0)
            act_hold_d    = d['action_hold']          - prev_d.get('action_hold',          0)
            act_close_d   = d['action_close']         - prev_d.get('action_close',         0)

            tw_total_w    = tw_delta + tl_delta

            # ── Metryki okna ──────────────────────────────────────────────────
            pnl_avg_w     = pnl_delta / ep_delta     if ep_delta > 0      else None
            wr_w          = tw_delta  / tw_total_w   if tw_total_w > 0    else None
            len_avg_w     = len_sum_d / len_cnt_d    if len_cnt_d > 0     else None
            # Profit Factor okna
            pf_w          = profit_d / loss_d        if loss_d > 1e-9     else (999.0 if profit_d > 0 else 0.0)
            # Win/Loss Ratio (wielkość): avg_win / avg_loss
            avg_win_w     = profit_d / tw_delta      if tw_delta > 0      else None
            avg_loss_w    = loss_d   / tl_delta      if tl_delta > 0      else None
            wl_ratio_w    = avg_win_w / avg_loss_w   if avg_win_w and avg_loss_w else None
            # Avg Hold Time Win vs Loss
            hold_win_w    = whs_d / tw_delta         if tw_delta > 0      else None
            hold_loss_w   = lhs_d / tl_delta         if tl_delta > 0      else None
            # Holding Time Efficiency: MFE / hold_steps (na transakcję)
            total_hold_d  = whs_d + lhs_d
            hte_w         = (mfe_d / tw_total_w) / (total_hold_d / tw_total_w) \
                            if tw_total_w > 0 and total_hold_d > 0 else None
            # Long/Short Bias
            total_opens_d = long_d + short_d
            long_bias_w   = long_d / total_opens_d   if total_opens_d > 0 else None
            # Action Distribution okna
            total_act_d   = act_long_d + act_short_d + act_hold_d + act_close_d
            act_dist_w    = {
                'long':  act_long_d  / total_act_d if total_act_d > 0 else 0.0,
                'short': act_short_d / total_act_d if total_act_d > 0 else 0.0,
                'hold':  act_hold_d  / total_act_d if total_act_d > 0 else 0.0,
                'close': act_close_d / total_act_d if total_act_d > 0 else 0.0,
            }
            # Slippage & Spread Impact (koszt prowizji vs net PnL w oknie)
            commission_w  = tr_delta * data['cost_per_trade']
            slippage_pct  = commission_w / abs(pnl_delta) if abs(pnl_delta) > 1e-9 else None

            # ── Metryki lifetime ──────────────────────────────────────────────
            ep_total      = d['episode_count']
            tot_trades    = d['episode_trade_wins'] + d['episode_trade_losses']
            tot_wr        = d['episode_trade_wins'] / tot_trades if tot_trades > 0 else None
            tot_pnl_avg   = d['episode_pnl_sum'] / ep_total     if ep_total > 0    else None
            tot_len_avg   = d['episode_length_sum'] / d['episode_length_count'] \
                            if d['episode_length_count'] > 0 else None
            pf_t          = d['profit_sum'] / d['loss_sum']      if d['loss_sum'] > 1e-9 \
                            else (999.0 if d['profit_sum'] > 0 else 0.0)
            avg_win_t     = d['profit_sum'] / d['episode_trade_wins']  if d['episode_trade_wins'] > 0 else None
            avg_loss_t    = d['loss_sum']   / d['episode_trade_losses'] if d['episode_trade_losses'] > 0 else None
            wl_ratio_t    = avg_win_t / avg_loss_t if avg_win_t and avg_loss_t else None
            hold_win_t    = d['win_hold_steps']  / d['episode_trade_wins']  if d['episode_trade_wins'] > 0 else None
            hold_loss_t   = d['loss_hold_steps'] / d['episode_trade_losses'] if d['episode_trade_losses'] > 0 else None
            total_opens_t = d['long_opens'] + d['short_opens']
            long_bias_t   = d['long_opens'] / total_opens_t if total_opens_t > 0 else None

            # Sharpe, Sortino, MaxDD z deque PnL epizodów
            pnl_deque     = list(d.get('episode_pnl_deque', []))
            sharpe, sortino = _sharpe_sortino(pnl_deque)
            max_dd        = _max_drawdown(pnl_deque)
            recovery      = d['episode_pnl_sum'] / max_dd if max_dd > 1e-9 else None

            actors[symbol] = {
                'window': {
                    'episodes':      ep_delta,
                    'pnl_avg':       pnl_avg_w,
                    'win_rate':      wr_w,
                    'avg_length':    len_avg_w,
                    'transactions':  tr_delta,
                    'trade_wins':    tw_delta,
                    'trade_losses':  tl_delta,
                    'profit_factor': pf_w,
                    'wl_ratio':      wl_ratio_w,
                    'avg_win':       avg_win_w,
                    'avg_loss':      avg_loss_w,
                    'hold_win':      hold_win_w,
                    'hold_loss':     hold_loss_w,
                    'hte':           hte_w,
                    'long_bias':     long_bias_w,
                    'act_dist':      act_dist_w,
                    'slippage_pct':  slippage_pct,
                    'commission':    commission_w,
                },
                'total': {
                    'episodes':      ep_total,
                    'pnl_avg':       tot_pnl_avg,
                    'win_rate':      tot_wr,
                    'avg_length':    tot_len_avg,
                    'transactions':  d['transactions'],
                    'max_consec_losses': d['max_consecutive_losses'],
                    'profit_factor': pf_t,
                    'wl_ratio':      wl_ratio_t,
                    'avg_win':       avg_win_t,
                    'avg_loss':      avg_loss_t,
                    'hold_win':      hold_win_t,
                    'hold_loss':     hold_loss_t,
                    'long_bias':     long_bias_t,
                    'sharpe':        sharpe,
                    'sortino':       sortino,
                    'max_dd':        max_dd,
                    'recovery':      recovery,
                },
                'pnl_arrow': _arrow(pnl_avg_w, None),
            }

        data['actors'] = actors

        # ── Neural metrics ────────────────────────────────────────────────────
        acc = trainer._metrics_accumulator
        n   = acc['steps_accumulated']
        data['policy_entropy'] = acc['action_entropy_sum'] / n if n > 0 else None
        data['q_hold_avg']     = acc['q_hold_sum']         / n if n > 0 else None

        # Config prowizji — do obliczenia kosztu na transakcję
        reward_cfg = trainer.config.get('reward', {})
        data['cost_per_trade'] = (
            reward_cfg.get('commission_open',  0.00075) +
            reward_cfg.get('commission_close', 0.00075) +
            reward_cfg.get('trade_penalty',    0.00050) +
            reward_cfg.get('close_penalty',    0.00050)
        )

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

                def _ratio(v):
                    return f'{v:.3f}' if v is not None else '—'
                def _steps(v):
                    return f'{v:.1f} kr' if v is not None else '—'
                def _pf(v):
                    return f'{v:.2f}' if v is not None and v < 900 else ('∞' if v else '—')

                lines += [
                    f'  <b>{sym}</b>',
                    f'  ↳ <i>Okno ({w["episodes"]} ep, {w["transactions"]} tr)</i>',
                    f'    PnL avg:       {_fmt_pnl(w["pnl_avg"])}',
                    f'    Win Rate:      {_pct(w["win_rate"])}',
                    f'    Profit Factor: {_pf(w["profit_factor"])}',
                    f'    Win/Loss Ratio:{_ratio(w["wl_ratio"])}',
                    f'    Avg win:       {_fmt_pnl(w["avg_win"])}',
                    f'    Avg loss:      {_fmt_pnl(w["avg_loss"])}' if w['avg_loss'] else '    Avg loss:      —',
                    f'    Hold Win:      {_steps(w["hold_win"])}',
                    f'    Hold Loss:     {_steps(w["hold_loss"])}',
                    f'    Dł. epizodu:   {w["avg_length"]:.1f} kr' if w['avg_length'] else '    Dł. epizodu:   —',
                    f'    HT Efficiency: {w["hte"]:.4f}' if w['hte'] is not None else '    HT Efficiency: —',
                    f'    Long bias:     {_pct(w["long_bias"])} L / {_pct(1 - w["long_bias"]) if w["long_bias"] is not None else "—"} S',
                    f'    Akcje:  H={w["act_dist"]["hold"]:.0%} L={w["act_dist"]["long"]:.0%} S={w["act_dist"]["short"]:.0%} C={w["act_dist"]["close"]:.0%}',
                    f'    Prowizje:      {w["commission"]:.4f}',
                    f'    Koszt/PnL:     {_pct(w["slippage_pct"])}' if w['slippage_pct'] is not None else '    Koszt/PnL:     —',
                    f'  ↳ <i>Łącznie ({t["episodes"]} ep, {t["transactions"]} tr)</i>',
                    f'    PnL avg:       {_fmt_pnl(t["pnl_avg"])}',
                    f'    Win Rate:      {_pct(t["win_rate"])}',
                    f'    Profit Factor: {_pf(t["profit_factor"])}',
                    f'    Win/Loss Ratio:{_ratio(t["wl_ratio"])}',
                    f'    Hold Win:      {_steps(t["hold_win"])}',
                    f'    Hold Loss:     {_steps(t["hold_loss"])}',
                    f'    Dł. epizodu:   {t["avg_length"]:.1f} kr' if t['avg_length'] else '    Dł. epizodu:   —',
                    f'    Long bias:     {_pct(t["long_bias"])} L / {_pct(1 - t["long_bias"]) if t["long_bias"] is not None else "—"} S',
                    f'    Sharpe:        {t["sharpe"]:.3f}'   if t['sharpe']   is not None else '    Sharpe:        —',
                    f'    Sortino:       {t["sortino"]:.3f}'  if t['sortino']  is not None else '    Sortino:       —',
                    f'    Max Drawdown:  {_fmt_pnl(t["max_dd"])}',
                    f'    Recovery:      {_ratio(t["recovery"])}',
                    f'    Max strat z rz:{t["max_consec_losses"]}',
                ]

        # ── Diagnostyka sieci ─────────────────────────────────────────────────
        lqs = data.get('live_q_spread')
        las = data.get('live_adv_std')
        ent = data.get('policy_entropy')
        qh  = data.get('q_hold_avg')
        lines += [
            '',
            '🧠 <b>Diagnostyka sieci</b> <i>(live)</i>',
            f'  Q-spread:       {lqs:.4f}' if lqs is not None else '  Q-spread:       —',
            f'  Q-Hold (anchor):{qh:.4f}'  if qh  is not None else '  Q-Hold (anchor):—',
            f'  Policy Entropy: {ent:.4f} bit' if ent is not None else '  Policy Entropy: —',
            f'  Adv std:        {las:.4f}' if las is not None else '  Adv std:        —',
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
