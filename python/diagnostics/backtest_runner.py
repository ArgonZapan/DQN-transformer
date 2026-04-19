"""
BacktestRunner — uruchamia backtest na danych walidacyjnych przed raportem Telegram.
Działa wyłącznie na CPU żeby nie zakłócać treningu na GPU.
"""
import glob
import logging
import os
import sys

import numpy as np

logger = logging.getLogger('learner')

_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


def _find_latest_checkpoint(config: dict) -> str | None:
    ckpt_dir = os.path.join(_PYTHON_DIR, 'checkpoints')
    candidates = []

    for name in ('shutdown_checkpoint.pt', 'hourly_checkpoint.pt'):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            candidates.append(p)

    step_pts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_step_*.pt')))
    candidates.extend(step_pts)

    if not candidates:
        return None

    # Wybierz ten z najwyższym step
    import torch
    best, best_step = None, -1
    for p in candidates:
        try:
            c = torch.load(p, map_location='cpu', weights_only=False)
            s = c.get('step', 0)
            if s > best_step:
                best_step, best = s, p
        except Exception:
            pass
    return best


def run_backtest(config: dict, checkpoint_path: str | None = None,
                 trainer=None) -> dict:
    """
    Uruchamia backtest na danych walidacyjnych dla wszystkich symboli.
    Jeśli podano trainer — kopiuje wagi z in-memory modelu (aktualny krok).
    Fallback: ładuje najnowszy checkpoint z dysku.
    """
    import copy
    import torch
    from backtest import load_model, run_symbol_backtest, calc_metrics
    from model.network import TradingDQN

    ckpt_label = 'in-memory'
    ckpt_step = 0

    if trainer is not None and trainer.step_count == 0:
        logger.info('[Backtest] Skipping — model not trained yet (step=0)')
        return {
            'checkpoint': 'step 0 — model not trained',
            'per_symbol': {},
            'combined': {},
            'skipped': True,
        }

    if trainer is not None:
        # Kopiuj wagi z GPU trenera → nowy model na CPU (nie zakłóca treningu)
        logger.info(f'[Backtest] Copying in-memory model (step={trainer.step_count}) to CPU...')
        model = TradingDQN(config).to('cpu')
        eager = getattr(trainer.main_network, '_orig_mod', trainer.main_network)
        model.load_state_dict(copy.deepcopy(eager.state_dict()))
        model.eval()
        ckpt_label = f'in-memory step {trainer.step_count:,}'
        ckpt_step = trainer.step_count
    else:
        if checkpoint_path is None:
            checkpoint_path = _find_latest_checkpoint(config)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError('Brak checkpointu do backtestowania')
        import torch as _torch
        _meta = _torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        ckpt_step = _meta.get('step', 0)
        ckpt_label = f'{os.path.basename(checkpoint_path)} (step {ckpt_step:,})'
        logger.info(f'[Backtest] Loading from disk: {ckpt_label}')
        model = load_model(checkpoint_path, config, device='cpu')

    symbols = [a['symbol'] for a in config['actors']]

    from backtest import preload_backtest_data
    preload_backtest_data(symbols, config)

    per_symbol, all_trades, all_equity, all_daily = {}, [], [0.0], {}

    for symbol in symbols:
        logger.info(f'[Backtest] Processing {symbol}...')
        try:
            res = run_symbol_backtest(symbol, config, model, device='cpu')
            m = calc_metrics(res['trades'], res['equity_curve'], res['daily_pnl'])
            trades = res['trades']
            long_count = sum(1 for t in trades if t.get('side') == 'LONG')
            long_bias = long_count / len(trades) if trades else None
            per_symbol[symbol] = {**m, 'total_steps': res['total_steps'], 'long_bias': long_bias}
            all_trades.extend(res['trades'])
            for t in res['trades']:
                all_equity.append(all_equity[-1] + t['pnl'])
            for d, v in res['daily_pnl'].items():
                all_daily[d] = all_daily.get(d, 0.0) + v
        except Exception as e:
            logger.warning(f'[Backtest] {symbol} failed: {e}')
            per_symbol[symbol] = None

    combined = calc_metrics(all_trades, all_equity, all_daily)
    logger.info(f'[Backtest] Done: {len(all_trades)} trades, combined PnL={combined["total_pnl"]:+.4f}')

    return {
        'checkpoint': ckpt_label,
        'per_symbol': per_symbol,
        'combined': combined,
    }


def format_backtest_telegram(result: dict) -> str:
    """Formatuje wyniki backtesta do wiadomości Telegram (HTML)."""
    if result.get('skipped'):
        return '\n\n📈 <b>Backtest (OOS)</b>\n  <i>Model niewyuczony — backtest pominięty.</i>'
    lines = ['', '📈 <b>Backtest (OOS)</b>', f'  Model: <code>{result["checkpoint"]}</code>']

    def _pct(v):
        return f'{v:.1%}' if v is not None else '—'

    def _pf(v):
        if v is None:
            return '—'
        return f'{v:.2f}' if v < 900 else '∞'

    def _pnl(v):
        if v is None:
            return '—'
        if abs(v) < 1e-9:
            return '0.0000%'
        sign = '+' if v > 0 else ''
        return f'{sign}{v:.4%}'

    per_symbol = result.get('per_symbol', {})
    for symbol, m in per_symbol.items():
        if m is None:
            lines += [f'  <b>{symbol}</b>: błąd danych']
            continue

        n = m['total_trades']
        pnl = m['total_pnl']
        wr = m['win_rate']
        sharpe = m.get('sharpe_ratio')
        dd = m.get('max_drawdown')
        pf = m.get('profit_factor')
        avg_w = m.get('avg_win')
        avg_l = m.get('avg_loss')
        mcl = m.get('max_consecutive_losses', 0)
        steps = m.get('total_steps', 0)

        long_bias = m.get('long_bias')
        long_bias_str = (f'{long_bias:.1%} L / {1 - long_bias:.1%} S' if long_bias is not None else '—')

        lines += [
            f'',
            f'  <b>{symbol}</b> <i>({steps:,} kroków)</i>',
            f'    Transakcji:  {n}',
            f'    PnL łącznie: {_pnl(pnl)}',
            f'    Win Rate:    {_pct(wr)}',
            f'    Profit Factor:{_pf(pf)}',
            f'    Avg win:     {_pnl(avg_w)}' if avg_w else '    Avg win:     —',
            f'    Avg loss:    {_pnl(-avg_l)}' if avg_l else '    Avg loss:    —',
            f'    Max DD:      {_pnl(-dd)}' if dd else '    Max DD:      —',
            f'    Sharpe ann:  {sharpe:.3f}' if sharpe else '    Sharpe ann:  —',
            f'    Long bias:   {long_bias_str}',
            f'    Max strat z rz:{mcl}',
        ]

    c = result.get('combined', {})
    if c and per_symbol:
        ct = c['total_trades']
        lines += [
            '',
            f'  <b>COMBINED</b>',
            f'    Transakcji:  {ct}',
            f'    PnL łącznie: {_pnl(c["total_pnl"])}',
            f'    Win Rate:    {_pct(c["win_rate"])}',
            f'    Profit Factor:{_pf(c["profit_factor"])}',
            f'    Sharpe ann:  {c["sharpe_ratio"]:.3f}',
            f'    Max DD:      {_pnl(-c["max_drawdown"])}',
        ]

    return '\n'.join(lines)
