"""
Raport diagnostyczny — CLI PASS/FAIL per metryka.

Użycie:
  python -m diagnostics.report
  python -m diagnostics.report --window 500
  python -m diagnostics.report --metrics path/to/metrics.jsonl
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

# Upewnij się że python/ jest w sys.path (gdy skrypt wywołany z ROOT przez run_report.bat)
_python_dir = os.path.join(os.path.dirname(__file__), '..')
if _python_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_python_dir))


# ── ANSI ──────────────────────────────────────────────────────────────────
def _green(s): return f'\x1b[32m{s}\x1b[0m'
def _yellow(s): return f'\x1b[33m{s}\x1b[0m'
def _red(s): return f'\x1b[31m{s}\x1b[0m'
def _bold(s): return f'\x1b[1m{s}\x1b[0m'
def _gray(s): return f'\x1b[2m{s}\x1b[0m'

STATUS_COLOR = {'PASS': _green, 'WARN': _yellow, 'FAIL': _red}


# ── thresholds ────────────────────────────────────────────────────────────
THRESHOLDS = {
    'q_spread':        {'pass': 0.05,  'warn': 0.02},
    'grad_norm_low':   {'pass': 0.005, 'warn': 0.002},   # dolna granica
    'grad_norm_high':  {'pass': 2.0,   'warn': 5.0},     # górna granica
    'advantage_std':   {'pass': 0.01,  'warn': 0.005},
    'dominant_ratio':  {'pass': 0.80,  'warn': 0.90},    # mniej = lepiej
    'td_std':          {'pass': 0.01,  'warn': 0.005},
    'action_div':      {'pass': 4,     'warn': 3},       # więcej = lepiej
}


def _load_jsonl(path: str, last_n: int) -> list:
    records = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return []
    return records[-last_n:]


def _safe_mean(vals: list) -> float | None:
    clean = [v for v in vals if v is not None and isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.mean(clean) if clean else None


def _trend(vals: list) -> str:
    if len(vals) < 10:
        return 'za mało danych'
    half = len(vals) // 2
    first = _safe_mean(vals[:half])
    second = _safe_mean(vals[half:])
    if first is None or second is None or first == 0:
        return '?'
    delta_pct = (second - first) / abs(first) * 100
    if delta_pct < -5:
        return _green(f'↓ {delta_pct:.1f}%')
    elif delta_pct > 5:
        return _red(f'↑ +{delta_pct:.1f}%')
    else:
        return _yellow(f'→ {delta_pct:+.1f}%')


def _verdict(value, metric: str) -> str:
    t = THRESHOLDS.get(metric)
    if t is None or value is None:
        return 'SKIP'
    # metryki gdzie mniej = lepiej
    if metric == 'dominant_ratio':
        if value <= t['pass']:
            return 'PASS'
        elif value <= t['warn']:
            return 'WARN'
        return 'FAIL'
    # metryki gdzie więcej = lepiej
    if value >= t['pass']:
        return 'PASS'
    elif value >= t['warn']:
        return 'WARN'
    return 'FAIL'


def _fmt_row(label: str, status: str, value_str: str, note: str = '') -> str:
    pad = 28
    color = STATUS_COLOR.get(status, lambda s: s)
    status_str = color(f'[{status}]')
    return f'  {status_str}  {label:<{pad}} {value_str}  {_gray(note)}'


def run_report(metrics_path: str, health_path: str, baseline_path: str,
               window: int = 1000) -> int:
    """Zwraca 0 jeśli PASS, 1 jeśli WARN, 2 jeśli FAIL."""

    records = _load_jsonl(metrics_path, window)
    health = _load_jsonl(health_path, 50)

    # Załaduj baseline
    baseline = {}
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
        except Exception:
            pass

    if not records:
        print(_red(f'Brak danych w {metrics_path}'))
        return 2

    step_first = records[0].get('step', 0)
    step_last = records[-1].get('step', 0)
    ts_last = records[-1].get('ts', 0)
    ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_last)) if ts_last else '?'

    print()
    print(_bold('=' * 66))
    print(_bold(f'  DQN-OPUS Diagnostic Report'))
    print(f'  Okno: ostatnie {len(records)} rekordów  |  kroki {step_first}–{step_last}')
    print(f'  Ostatni log: {ts_str}')
    print(_bold('=' * 66))

    verdicts = []

    # ── Loss ────────────────────────────────────────────────────────────
    losses = [r.get('loss') for r in records]
    loss_last = _safe_mean(losses[-20:])
    loss_trend = _trend(losses)
    print(f'\n  {_bold("TRENING")}')
    status = 'PASS' if loss_last is not None and loss_last < 0.01 else 'WARN'
    verdicts.append(status)
    val = f'{loss_last:.5f}' if loss_last is not None else '?'
    print(_fmt_row('Loss (ostatnie 20)', status, val, f'trend: {loss_trend}'))

    # ── Gradient norm ────────────────────────────────────────────────────
    grads = [r.get('grad_norm') for r in records]
    grad_last = _safe_mean(grads[-20:])
    if grad_last is not None:
        if grad_last < THRESHOLDS['grad_norm_low']['warn']:
            s = 'FAIL'
        elif grad_last < THRESHOLDS['grad_norm_low']['pass']:
            s = 'WARN'
        elif grad_last > THRESHOLDS['grad_norm_high']['warn']:
            s = 'FAIL'
        elif grad_last > THRESHOLDS['grad_norm_high']['pass']:
            s = 'WARN'
        else:
            s = 'PASS'
        verdicts.append(s)
        note = 'zanikające' if grad_last < 0.005 else ('eksplodujące' if grad_last > 5.0 else 'OK')
        print(_fmt_row('Gradient norm', s, f'{grad_last:.5f}', f'zakres 0.005–2.0 | {note}'))

    # ── Q-spread ─────────────────────────────────────────────────────────
    print(f'\n  {_bold("Q-WARTOŚCI")}')
    spreads = [r.get('q_spread') for r in records]
    spread_last = _safe_mean(spreads[-20:])
    s = _verdict(spread_last, 'q_spread')
    verdicts.append(s)
    spread_trend = _trend(spreads)
    val = f'{spread_last:.4f}' if spread_last is not None else '?'
    note = f'próg PASS: {THRESHOLDS["q_spread"]["pass"]} | trend: {spread_trend}'
    print(_fmt_row('Q-spread', s, val, note))

    # vs baseline
    if baseline.get('q_spread_mean') and spread_last is not None:
        ratio = spread_last / baseline['q_spread_mean']
        bs = 'PASS' if ratio >= 2.0 else ('WARN' if ratio >= 1.0 else 'FAIL')
        verdicts.append(bs)
        print(_fmt_row('Q-spread vs baseline', bs, f'{ratio:.2f}×',
                       f'random={baseline["q_spread_mean"]:.4f}'))

    # ── Advantage std ─────────────────────────────────────────────────────
    adv_stds = [r.get('advantage_std') for r in records]
    adv_last = _safe_mean([v for v in adv_stds[-50:] if v is not None])
    if adv_last is not None:
        s = _verdict(adv_last, 'advantage_std')
        verdicts.append(s)
        print(_fmt_row('Advantage std', s, f'{adv_last:.5f}',
                       f'próg PASS: {THRESHOLDS["advantage_std"]["pass"]}'))

    # ── TD error std ─────────────────────────────────────────────────────
    td_stds = [r.get('td_std') for r in records]
    td_last = _safe_mean(td_stds[-20:])
    if td_last is not None:
        s = _verdict(td_last, 'td_std')
        verdicts.append(s)
        print(_fmt_row('TD error std', s, f'{td_last:.5f}',
                       f'próg PASS: {THRESHOLDS["td_std"]["pass"]}'))

    # ── Rozkład akcji ─────────────────────────────────────────────────────
    print(f'\n  {_bold("POLITYKA")}')
    action_names = ['LONG', 'SHORT', 'HOLD', 'CLOSE']
    all_ratios = [r.get('action_ratios') for r in records if r.get('action_ratios')]
    if all_ratios:
        last_ratios = _safe_mean([r[-len(all_ratios)//5:] for r in zip(*all_ratios)] if len(all_ratios) > 5 else [all_ratios[-1]])
        # Policz osobno dla każdej akcji
        for i, name in enumerate(action_names):
            vals = [r[i] for r in all_ratios if len(r) > i]
            mean_val = _safe_mean(vals[-20:])
            if mean_val is not None:
                marker = ' ← dominant' if mean_val == max(_safe_mean([r[j] for r in all_ratios[-20:] if len(r) > j] or [0]) or 0 for j in range(4)) else ''
                print(f'  {_gray("      ")}  {name:<6}  {mean_val:.1%}{marker}')

        # Dominant ratio
        dominant_ratios = []
        for r in all_ratios[-20:]:
            if r:
                dominant_ratios.append(max(r))
        dom_last = _safe_mean(dominant_ratios)
        if dom_last is not None:
            s = _verdict(dom_last, 'dominant_ratio')
            verdicts.append(s)
            print(_fmt_row('Dominant action ratio', s, f'{dom_last:.1%}',
                           f'próg FAIL: >{THRESHOLDS["dominant_ratio"]["warn"]:.0%}'))

    # ── Health checks ─────────────────────────────────────────────────────
    if health:
        print(f'\n  {_bold("HEALTH CHECKS")} ({len(health)} checkpointów)')
        last_h = health[-1]
        div = last_h.get('action_diversity')
        if div is not None:
            s = _verdict(div, 'action_div')
            verdicts.append(s)
            print(_fmt_row('Action diversity', s, f'{div}/4 akcji',
                           f'dominant: {last_h.get("dominant_action","?")}@{last_h.get("dominant_ratio",0):.1%}'))
        h_spread = last_h.get('q_spread_mean')
        if h_spread is not None:
            s = _verdict(h_spread, 'q_spread')
            print(_fmt_row('Health q_spread', s, f'{h_spread:.4f}', '(100 stanów z buffera)'))

    # ── Werdykt końcowy ───────────────────────────────────────────────────
    n_fail = verdicts.count('FAIL')
    n_warn = verdicts.count('WARN')
    n_pass = verdicts.count('PASS')

    print()
    print(_bold('=' * 66))
    if n_fail > 0:
        overall = _red(f'FAIL  ({n_fail} krytycznych, {n_warn} ostrzeżeń, {n_pass} OK)')
        exit_code = 2
    elif n_warn > 0:
        overall = _yellow(f'WARN  ({n_warn} ostrzeżeń, {n_pass} OK)')
        exit_code = 1
    else:
        overall = _green(f'PASS  (wszystkie {n_pass} metryk OK)')
        exit_code = 0
    print(f'  WERDYKT: {overall}')

    # Rekomendacje
    recs = _recommendations(records, health, baseline)
    if recs:
        print(f'\n  {_bold("REKOMENDACJE:")}')
        for r in recs:
            print(f'  • {r}')

    print(_bold('=' * 66))
    print()
    return exit_code


def _recommendations(records: list, health: list, baseline: dict) -> list:
    recs = []
    spreads = [r.get('q_spread') for r in records if r.get('q_spread')]
    if spreads and _safe_mean(spreads[-20:]) < 0.02:
        recs.append('Q-spread bardzo niski — rozważ wyższy lr lub niższy dropout')
    grads = [r.get('grad_norm') for r in records if r.get('grad_norm')]
    if grads:
        g = _safe_mean(grads[-20:])
        if g and g < 0.005:
            recs.append('Gradienty zanikają — sieć konwerguje do trivialnego rozwiązania, sprawdź reward shaping')
        if g and g > 2.0:
            recs.append('Gradienty duże — rozważ obniżenie lr lub zwiększenie grad clip threshold')
    adv = [r.get('advantage_std') for r in records if r.get('advantage_std')]
    if adv and _safe_mean(adv[-50:]) < 0.005:
        recs.append('Advantage stream umiera — dueling head traci zdolność różnicowania akcji')
    if health:
        last_h = health[-1]
        if last_h.get('action_diversity', 4) <= 2:
            recs.append(f'Sieć wybiera tylko {last_h["action_diversity"]}/4 akcji — eksploracja zbyt mała lub reward zbyt jednorodny')
    return recs


def main():
    parser = argparse.ArgumentParser(description='DQN-OPUS Diagnostic Report')
    parser.add_argument('--window', type=int, default=1000,
                        help='Liczba ostatnich rekordów do analizy (domyślnie 1000)')
    parser.add_argument('--metrics', default='python/diagnostics/metrics.jsonl')
    parser.add_argument('--health', default='python/diagnostics/health_checks.jsonl')
    parser.add_argument('--baseline', default='python/diagnostics/baseline.json')
    args = parser.parse_args()

    exit_code = run_report(args.metrics, args.health, args.baseline, args.window)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
