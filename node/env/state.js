const { calculateRSI, calculateMACD, calculateSMA, rollingMean, rollingStd } = require('../data/indicators');

const TIMEFRAME_KEYS = ['1m', '15m', '1h', '1d', '1w'];
const TIMEFRAME_CONFIG_KEYS = {
    '1m': 'candles_1m',
    '15m': 'candles_15m',
    '1h': 'candles_1h',
    '1d': 'candles_1d',
    '1w': 'candles_1w'
};
const NUM_FEATURES = 8;

function buildFeatures(candles, normWindow) {
    if (candles.length === 0) return [];

    const closes = candles.map(c => c.close);
    const opens = candles.map(c => c.open);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const volumes = candles.map(c => c.volume);

    const mean = rollingMean(closes, normWindow);
    const std = rollingStd(closes, normWindow);
    const rsi = calculateRSI(closes);
    const { macdLine } = calculateMACD(closes);
    const sma20 = calculateSMA(closes, 20);
    const meanVolume = rollingMean(volumes, normWindow);

    const features = [];

    for (let i = 0; i < candles.length; i++) {
        const c = closes[i];
        const o = opens[i];
        const h = highs[i];
        const l = lows[i];
        const v = volumes[i];

        const normalizedClose = std[i] !== 0 ? (c - mean[i]) / std[i] : 0;
        const relativeRange = c !== 0 ? (h - l) / c : 0;
        const candleDirection = c !== 0 ? (c - o) / c : 0;
        const normalizedVolume = meanVolume[i] !== 0 ? v / meanVolume[i] : 0;
        const rsiNorm = rsi[i] / 100;
        const macdNorm = c !== 0 ? macdLine[i] / c : 0;
        const prevClose = i > 0 ? closes[i - 1] : c;
        const pctChange = prevClose !== 0 ? (c - prevClose) / prevClose : 0;
        const aboveSma = c > sma20[i] ? 1.0 : 0.0;

        features.push([
            normalizedClose,
            relativeRange,
            candleDirection,
            normalizedVolume,
            rsiNorm,
            macdNorm,
            pctChange,
            aboveSma
        ]);
    }

    return features;
}

/**
 * Synchronizacja czasowa — użyj tylko zamkniętych świec
 * aby uniknąć look-ahead bias.
 * Binary search O(log N) zamiast filter O(N).
 */
function getAlignedCandles(allCandles, currentTime, timeframe, numCandles) {
    let lo = 0, hi = allCandles.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (allCandles[mid].close_time <= currentTime) lo = mid + 1;
        else hi = mid;
    }
    const start = numCandles > 0 ? Math.max(0, lo - numCandles) : 0;
    return allCandles.slice(start, lo);
}

function buildState(allCandlesPerTf, currentTime, config) {
    const timeframesConfig = config.timeframes;
    const normWindow = config.data.normalization_window;
    const state = {};

    for (const tf of TIMEFRAME_KEYS) {
        const configKey = TIMEFRAME_CONFIG_KEYS[tf];
        const numCandles = timeframesConfig[configKey];
        if (!numCandles || numCandles <= 0) continue;

        const candles = allCandlesPerTf[tf] || [];
        const aligned = getAlignedCandles(candles, currentTime, tf, numCandles);

        if (aligned.length < numCandles) {
            const features = buildFeatures(aligned, normWindow);
            const padding = numCandles - aligned.length;
            const paddedFeatures = [];
            for (let i = 0; i < padding; i++) paddedFeatures.push([0, 0, 0, 0, 0, 0, 0, 0]);
            state[tf] = paddedFeatures.concat(features);
        } else {
            state[tf] = buildFeatures(aligned, normWindow);
        }
    }

    return state;
}

/**
 * Pre-oblicza wszystkie wskaźniki dla pełnej tablicy świec danego TF.
 * Wynik: 8 tablic Float64Array indeksowanych pozycją świecy.
 * Wykonywane raz po załadowaniu danych — O(N) zamiast O(N) na każdym kroku.
 */
function precomputeIndicators(candles, normWindow) {
    const n = candles.length;
    if (n === 0) return null;

    const closes  = new Float64Array(n);
    const opens   = new Float64Array(n);
    const highs   = new Float64Array(n);
    const lows    = new Float64Array(n);
    const volumes = new Float64Array(n);

    for (let i = 0; i < n; i++) {
        closes[i]  = candles[i].close;
        opens[i]   = candles[i].open;
        highs[i]   = candles[i].high;
        lows[i]    = candles[i].low;
        volumes[i] = candles[i].volume;
    }

    const mean       = rollingMean(Array.from(closes), normWindow);
    const std        = rollingStd(Array.from(closes), normWindow);
    const rsi        = calculateRSI(Array.from(closes));
    const { macdLine } = calculateMACD(Array.from(closes));
    const sma20      = calculateSMA(Array.from(closes), 20);
    const meanVolume = rollingMean(Array.from(volumes), normWindow);

    const f = [
        new Float64Array(n), // 0: normalizedClose
        new Float64Array(n), // 1: relativeRange
        new Float64Array(n), // 2: candleDirection
        new Float64Array(n), // 3: normalizedVolume
        new Float64Array(n), // 4: rsiNorm
        new Float64Array(n), // 5: macdNorm
        new Float64Array(n), // 6: pctChange
        new Float64Array(n), // 7: aboveSma
    ];

    for (let i = 0; i < n; i++) {
        const c = closes[i], o = opens[i], h = highs[i], l = lows[i], v = volumes[i];
        f[0][i] = std[i]        !== 0 ? (c - mean[i]) / std[i]        : 0;
        f[1][i] = c             !== 0 ? (h - l) / c                   : 0;
        f[2][i] = c             !== 0 ? (c - o) / c                   : 0;
        f[3][i] = meanVolume[i] !== 0 ? v / meanVolume[i]             : 0;
        f[4][i] = rsi[i] / 100;
        f[5][i] = c             !== 0 ? macdLine[i] / c               : 0;
        f[6][i] = i > 0 && closes[i - 1] !== 0 ? (c - closes[i - 1]) / closes[i - 1] : 0;
        f[7][i] = c > sma20[i] ? 1.0 : 0.0;
    }

    return f;
}

/**
 * Składa macierz cech [numCandles × NUM_FEATURES] z pre-obliczonych tablic.
 * startIdx..endIdx to zakres w oryginalnej tablicy świec TF.
 */
function assembleFeaturesFromPrecomputed(f, startIdx, endIdx, numCandles) {
    const actualLen = endIdx - startIdx;
    const padding   = numCandles - actualLen;
    const result    = new Array(numCandles);

    for (let i = 0; i < padding; i++) {
        result[i] = [0, 0, 0, 0, 0, 0, 0, 0];
    }
    for (let i = 0; i < actualLen; i++) {
        const si = startIdx + i;
        result[padding + i] = [f[0][si], f[1][si], f[2][si], f[3][si], f[4][si], f[5][si], f[6][si], f[7][si]];
    }

    return result;
}

/**
 * Szybki buildState korzystający z pre-obliczonych wskaźników.
 * O(numCandles) zamiast O(N_total) per krok.
 */
function buildStateFromPrecomputed(precomputed, tfAlignments, stepIndex, config) {
    const state = {};

    for (const tf of TIMEFRAME_KEYS) {
        const configKey = TIMEFRAME_CONFIG_KEYS[tf];
        const numCandles = config.timeframes[configKey];
        if (!numCandles || numCandles <= 0) continue;
        if (!precomputed[tf]) continue;

        // Dla 1m endIdx = stepIndex + 1 (bieżąca świeca już zamknięta)
        const endIdx   = tf === '1m' ? stepIndex + 1 : tfAlignments[tf][stepIndex];
        const startIdx = Math.max(0, endIdx - numCandles);
        state[tf] = assembleFeaturesFromPrecomputed(precomputed[tf], startIdx, endIdx, numCandles);
    }

    return state;
}

function getActionMask(position) {
    if (!position || position.side === null) {
        return [1, 1, 1, 0];
    }
    return [0, 0, 1, 1];
}

module.exports = {
    buildState,
    buildFeatures,
    getAlignedCandles,
    getActionMask,
    precomputeIndicators,
    buildStateFromPrecomputed,
    TIMEFRAME_KEYS,
    TIMEFRAME_CONFIG_KEYS,
    NUM_FEATURES,
};
