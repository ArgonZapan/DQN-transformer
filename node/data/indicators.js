/**
 * Obliczanie wskaźników technicznych: RSI, MACD, SMA
 * Wartości są obliczane na tablicach świec (close prices).
 */

function calculateRSI(closes, period = 14) {
    const rsi = new Array(closes.length).fill(0);

    if (closes.length < period + 1) return rsi;

    let avgGain = 0;
    let avgLoss = 0;

    for (let i = 1; i <= period; i++) {
        const delta = closes[i] - closes[i - 1];
        if (delta > 0) avgGain += delta;
        else avgLoss += Math.abs(delta);
    }

    avgGain /= period;
    avgLoss /= period;

    if (avgLoss === 0) {
        rsi[period] = avgGain === 0 ? 50 : 100;
    } else {
        const rs = avgGain / avgLoss;
        rsi[period] = 100 - (100 / (1 + rs));
    }

    for (let i = period + 1; i < closes.length; i++) {
        const delta = closes[i] - closes[i - 1];
        const gain = delta > 0 ? delta : 0;
        const loss = delta < 0 ? Math.abs(delta) : 0;

        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;

        if (avgLoss === 0) {
            rsi[i] = avgGain === 0 ? 50 : 100;
        } else {
            const rs = avgGain / avgLoss;
            rsi[i] = 100 - (100 / (1 + rs));
        }
    }

    return rsi;
}

function calculateEMA(values, span) {
    const ema = new Array(values.length).fill(0);
    const multiplier = 2 / (span + 1);

    ema[0] = values[0];
    for (let i = 1; i < values.length; i++) {
        ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    return ema;
}

function calculateMACD(closes, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const ema12 = calculateEMA(closes, fastPeriod);
    const ema26 = calculateEMA(closes, slowPeriod);

    const macdLine = closes.map((_, i) => ema12[i] - ema26[i]);
    const signalLine = calculateEMA(macdLine, signalPeriod);
    const histogram = macdLine.map((v, i) => v - signalLine[i]);

    return { macdLine, signalLine, histogram };
}

function calculateSMA(values, period = 20) {
    const sma = new Array(values.length).fill(0);

    for (let i = 0; i < values.length; i++) {
        if (i < period - 1) {
            const slice = values.slice(0, i + 1);
            sma[i] = slice.reduce((a, b) => a + b, 0) / slice.length;
        } else {
            const slice = values.slice(i - period + 1, i + 1);
            sma[i] = slice.reduce((a, b) => a + b, 0) / period;
        }
    }

    return sma;
}

function rollingMean(values, window) {
    return calculateSMA(values, window);
}

function rollingStd(values, window) {
    const std = new Array(values.length).fill(0);
    // Algorytm Welforda O(n) — bieżąca suma i suma kwadratów w oknie przesuwnym
    let sumX = 0, sumX2 = 0;

    for (let i = 0; i < values.length; i++) {
        sumX  += values[i];
        sumX2 += values[i] * values[i];

        if (i >= window) {
            sumX  -= values[i - window];
            sumX2 -= values[i - window] * values[i - window];
        }

        const n = Math.min(i + 1, window);
        if (n < 2) { std[i] = 0; continue; }

        const mean = sumX / n;
        const variance = sumX2 / n - mean * mean;
        std[i] = variance > 0 ? Math.sqrt(variance) : 0;
    }

    return std;
}

// ── Wskaźniki v2 (niezaktywowane) ─────────────────────────────────────────────

/**
 * ATR (Average True Range) — volatility unormowana ceną.
 * TR = max(high-low, |high-prevClose|, |low-prevClose|)
 * ATR = EMA(TR, period)
 */
function calculateATR(highs, lows, closes, period = 14) {
    const n = closes.length;
    const tr = new Array(n).fill(0);
    tr[0] = highs[0] - lows[0];
    for (let i = 1; i < n; i++) {
        const hl = highs[i] - lows[i];
        const hc = Math.abs(highs[i] - closes[i - 1]);
        const lc = Math.abs(lows[i] - closes[i - 1]);
        tr[i] = Math.max(hl, hc, lc);
    }
    return calculateEMA(tr, period);
}

/**
 * Bollinger Band Width = (upper - lower) / sma = 4 * std20 / sma20.
 * Mierzy "ściskanie" pasma — spike poprzedza wybicie.
 * Zwraca tablicę bbWidth[i].
 */
function calculateBollingerWidth(closes, period = 20) {
    const sma = calculateSMA(closes, period);
    const std = rollingStd(closes, period);
    const bbw = new Array(closes.length).fill(0);
    for (let i = 0; i < closes.length; i++) {
        bbw[i] = sma[i] !== 0 ? (4 * std[i]) / sma[i] : 0;
    }
    return bbw;
}

/**
 * Stochastic %K = (close - min14) / (max14 - min14).
 * Mierzy pozycję ceny w 14-barowym zakresie H-L.
 */
function calculateStochasticK(highs, lows, closes, period = 14) {
    const n = closes.length;
    const stoch = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        const start = Math.max(0, i - period + 1);
        let minL = lows[start], maxH = highs[start];
        for (let j = start + 1; j <= i; j++) {
            if (lows[j] < minL) minL = lows[j];
            if (highs[j] > maxH) maxH = highs[j];
        }
        const range = maxH - minL;
        stoch[i] = range !== 0 ? (closes[i] - minL) / range : 0.5;
    }
    return stoch;
}

module.exports = {
    calculateRSI,
    calculateEMA,
    calculateMACD,
    calculateSMA,
    rollingMean,
    rollingStd,
    // v2
    calculateATR,
    calculateBollingerWidth,
    calculateStochasticK,
};
