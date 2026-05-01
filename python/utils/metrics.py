import math


def sharpe_ratio(returns, risk_free_rate=0.0):
    if not returns or len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1))
    if std_return == 0:
        return 0.0
    return (mean_return - risk_free_rate) / std_return


def max_drawdown(equity_curve):
    """Max drawdown as absolute value (not %) — peak minus trough in PnL units."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = peak - value          # absolute drop from peak (in equity units)
        if dd > max_dd:
            max_dd = dd
    return -max_dd


def win_rate(trades):
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return wins / len(trades)


def profit_factor(trades):
    gains = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    losses = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def avg_win_loss(trades):
    wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
    losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
    return {
        'avg_win': sum(wins) / len(wins) if wins else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
    }


def max_consecutive_losses(trades):
    max_streak = 0
    current_streak = 0
    for t in trades:
        if t.get('pnl', 0) < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
