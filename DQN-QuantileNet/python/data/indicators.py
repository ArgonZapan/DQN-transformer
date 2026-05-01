import numpy as np


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr)
    cumsum = np.cumsum(arr)
    out[0] = arr[0]
    # Leading values use expanding mean (window grows from 1 up to `window`).
    for i in range(1, min(window, len(arr))):
        out[i] = cumsum[i] / (i + 1)
    # Main rolling segment.
    for i in range(window, len(arr)):
        out[i] = (cumsum[i] - cumsum[i - window]) / window
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(arr))
    for i in range(1, len(arr)):
        start = max(0, i + 1 - window)
        out[i] = arr[start:i + 1].std()
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(closes, prepend=closes[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    if len(closes) > period:
        avg_gain[period] = gain[1:period + 1].mean()
        avg_loss[period] = loss[1:period + 1].mean()
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
        return np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(len(arr))
    k   = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = ema(closes, fast)
    ema_slow    = ema(closes, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.zeros(len(arr))
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        if i < period:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - period]) / period
    return out


def stochastic_k(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.zeros(len(closes))
    for i in range(len(closes)):
        start = max(0, i - period + 1)
        h = highs[start:i + 1].max()
        l = lows[start:i + 1].min()
        out[i] = (closes[i] - l) / (h - l) if h != l else 0.0
    return out


def compute_features(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, volumes: np.ndarray,
                     norm_window: int = 60) -> np.ndarray:
    """Compute 11 features. Returns float32 array of shape [N, 11].

    Division by zero is handled per-feature via np.where masks; np.errstate
    is scoped locally so importing this module does not silently disable
    NaN/inf warnings everywhere else (training, backtest, live loop).
    """
    n = len(closes)

    mean_c        = rolling_mean(closes, norm_window)
    std_c         = rolling_std(closes, norm_window)
    mean_v        = rolling_mean(volumes, norm_window)
    rsi_arr       = rsi(closes)
    stoch         = stochastic_k(highs, lows, closes)
    macd_l, sig_l = macd(closes)
    sma20         = sma(closes, 20)
    std20         = rolling_std(closes, 20)

    prev_closes = np.empty(n)
    prev_closes[0] = closes[0]
    prev_closes[1:] = closes[:-1]

    f = np.zeros((n, 11), dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        f[:, 0]  = np.where(std_c      != 0, (closes - mean_c) / std_c,         0.0)  # normalizedClose
        f[:, 1]  = np.where(closes     != 0, (highs - lows)    / closes,        0.0)  # relativeRange
        f[:, 2]  = np.where(closes     != 0, (closes - opens)  / closes,        0.0)  # candleDirection
        f[:, 3]  = np.where(mean_v     != 0, np.minimum(volumes / mean_v, 3.0), 0.0)  # volumeClipped
        f[:, 4]  = rsi_arr / 100.0                                                    # rsiNorm
        f[:, 5]  = stoch                                                              # stochasticK
        f[:, 6]  = np.where(closes     != 0, macd_l            / closes,        0.0)  # macdNorm
        f[:, 7]  = np.where(closes     != 0, (macd_l - sig_l)  / closes,        0.0)  # macdHistNorm
        f[:, 8]  = np.where(prev_closes != 0, (closes - prev_closes) / prev_closes, 0.0)  # pctChange
        f[:, 9]  = np.where(sma20      != 0, (4 * std20)       / sma20,         0.0)  # bollingerWidth
        f[:, 10] = np.where(closes     != 0, (closes - sma20)  / closes,        0.0)  # smaDistance

    return f
