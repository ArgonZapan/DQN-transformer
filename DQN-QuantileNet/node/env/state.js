const { calculateRSI, calculateMACD, calculateSMA, rollingMean, rollingStd,
        calculateATR, calculateBollingerWidth, calculateStochasticK } = require('../data/indicators');

const TIMEFRAME_KEYS = ['1m', '15m', '1h', '1d', '1w'];
const TIMEFRAME_CONFIG_KEYS = {
    '1m': 'candles_1m',
    '15m': 'candles_15m',
    '1h': 'candles_1h',
    '1d': 'candles_1d',
    '1w': 'candles_1w'
};
const NUM_FEATURES = 11;

// 11 features (v1+v2 merged, no duplicates):
//   0: normalizedClose   — z-score close (normWindow rolling window)
//   1: relativeRange     — (H-L)/close
//   2: candleDirection   — (close-open)/close
//   3: volumeClipped     — min(vol/meanVol, 3.0)
//   4: rsiNorm           — RSI(14)/100
//   5: stochasticK       — (close-min14)/(max14-min14)
//   6: macdNorm          — macdLine/close
//   7: macdHistNorm      — (macdLine-signal)/close
//   8: pctChange         — (close-prevClose)/prevClose
//   9: bollingerWidth    — 4*std20/sma20
//  10: smaDistance       — (close-sma20)/close

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
    const { macdLine, signalLine } = calculateMACD(closes);
    const sma20 = calculateSMA(closes, 20);
    const std20 = rollingStd(closes, 20);
    const meanVolume = rollingMean(volumes, normWindow);
    const stochK = calculateStochasticK(Array.from(highs), Array.from(lows), Array.from(closes));

    const features = [];

    for (let i = 0; i < candles.length; i++) {
        const c = closes[i];
        const o = opens[i];
        const h = highs[i];
        const l = lows[i];
        const v = volumes[i];
        const prevClose = i > 0 ? closes[i - 1] : c;

        features.push([
            std[i]        !== 0 ? (c - mean[i]) / std[i]                    : 0, // 0 normalizedClose
            c             !== 0 ? (h - l) / c                               : 0, // 1 relativeRange
            c             !== 0 ? (c - o) / c                               : 0, // 2 candleDirection
            meanVolume[i] !== 0 ? Math.min(v / meanVolume[i], 3.0)          : 0, // 3 volumeClipped
            rsi[i] / 100,                                                         // 4 rsiNorm
            stochK[i],                                                            // 5 stochasticK
            c             !== 0 ? macdLine[i] / c                           : 0, // 6 macdNorm
            c             !== 0 ? (macdLine[i] - signalLine[i]) / c         : 0, // 7 macdHistNorm
            prevClose     !== 0 ? (c - prevClose) / prevClose               : 0, // 8 pctChange
            sma20[i]      !== 0 ? (4 * std20[i]) / sma20[i]                : 0, // 9 bollingerWidth
            c             !== 0 ? (c - sma20[i]) / c                        : 0, // 10 smaDistance
        ]);
    }

    return features;
}

/**
 * Time alignment — use only closed candles to avoid look-ahead bias.
 * Binary search O(log N) instead of filter O(N).
 */
function getAlignedCandles(allCandles, currentTime, numCandles) {
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
        const aligned = getAlignedCandles(candles, currentTime, numCandles);

        if (aligned.length < numCandles) {
            const features = buildFeatures(aligned, normWindow);
            const padding = numCandles - aligned.length;
            const paddedFeatures = [];
            for (let i = 0; i < padding; i++) paddedFeatures.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            state[tf] = paddedFeatures.concat(features);
        } else {
            state[tf] = buildFeatures(aligned, normWindow);
        }
    }

    return state;
}

/**
 * Pre-computes all indicators for the full candle array of a given timeframe.
 * Result: 11 Float64Arrays indexed by candle position.
 * Executed once after data load — O(N) total instead of O(N) per step.
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
    const { macdLine, signalLine } = calculateMACD(Array.from(closes));
    const sma20      = calculateSMA(Array.from(closes), 20);
    const std20      = rollingStd(Array.from(closes), 20);
    const meanVolume = rollingMean(Array.from(volumes), normWindow);
    const stochK     = calculateStochasticK(Array.from(highs), Array.from(lows), Array.from(closes));

    const f = [
        new Float64Array(n), //  0: normalizedClose
        new Float64Array(n), //  1: relativeRange
        new Float64Array(n), //  2: candleDirection
        new Float64Array(n), //  3: volumeClipped
        new Float64Array(n), //  4: rsiNorm
        new Float64Array(n), //  5: stochasticK
        new Float64Array(n), //  6: macdNorm
        new Float64Array(n), //  7: macdHistNorm
        new Float64Array(n), //  8: pctChange
        new Float64Array(n), //  9: bollingerWidth
        new Float64Array(n), // 10: smaDistance
    ];

    for (let i = 0; i < n; i++) {
        const c = closes[i], o = opens[i], h = highs[i], l = lows[i], v = volumes[i];
        f[0][i]  = std[i]        !== 0 ? (c - mean[i]) / std[i]                : 0;
        f[1][i]  = c             !== 0 ? (h - l) / c                           : 0;
        f[2][i]  = c             !== 0 ? (c - o) / c                           : 0;
        f[3][i]  = meanVolume[i] !== 0 ? Math.min(v / meanVolume[i], 3.0)      : 0;
        f[4][i]  = rsi[i] / 100;
        f[5][i]  = stochK[i];
        f[6][i]  = c             !== 0 ? macdLine[i] / c                       : 0;
        f[7][i]  = c             !== 0 ? (macdLine[i] - signalLine[i]) / c     : 0;
        f[8][i]  = i > 0 && closes[i - 1] !== 0 ? (c - closes[i - 1]) / closes[i - 1] : 0;
        f[9][i]  = sma20[i]      !== 0 ? (4 * std20[i]) / sma20[i]            : 0;
        f[10][i] = c             !== 0 ? (c - sma20[i]) / c                    : 0;
    }

    return f;
}

/**
 * Assembles feature matrix [numCandles × NUM_FEATURES] from pre-computed arrays.
 * startIdx..endIdx is the range within the original TF candle array.
 */
function assembleFeaturesFromPrecomputed(f, startIdx, endIdx, numCandles) {
    const actualLen = endIdx - startIdx;
    const padding   = numCandles - actualLen;
    const result    = new Array(numCandles);

    for (let i = 0; i < padding; i++) {
        result[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    for (let i = 0; i < actualLen; i++) {
        const si = startIdx + i;
        result[padding + i] = [f[0][si], f[1][si], f[2][si], f[3][si], f[4][si], f[5][si],
                               f[6][si], f[7][si], f[8][si], f[9][si], f[10][si]];
    }

    return result;
}

/**
 * Fast buildState using pre-computed indicators.
 * O(numCandles) instead of O(N_total) per step.
 */
function buildStateFromPrecomputed(precomputed, tfAlignments, stepIndex, config) {
    const state = {};

    for (const tf of TIMEFRAME_KEYS) {
        const configKey = TIMEFRAME_CONFIG_KEYS[tf];
        const numCandles = config.timeframes[configKey];
        if (!numCandles || numCandles <= 0) continue;
        if (!precomputed[tf]) continue;

        // For 1m endIdx = stepIndex + 1 (current candle is already closed)
        const endIdx   = tf === '1m' ? stepIndex + 1 : tfAlignments[tf][stepIndex];
        const startIdx = Math.max(0, endIdx - numCandles);
        state[tf] = assembleFeaturesFromPrecomputed(precomputed[tf], startIdx, endIdx, numCandles);
    }

    return state;
}

// ── Indicators v2 (inactive) ──────────────────────────────────────────────────
//
// 8 features (compatible with num_features = 8, no model changes needed):
//   0: normalizedClose        — z-score with window 60 (unchanged)
//   1: relativeRange          — (H-L)/close (unchanged)
//   2: candleDirection        — (close-open)/close (unchanged)
//   3: volumeClipped          — min(vol/meanVol, 3.0) — capped at 3x mean instead of raw ratio
//   4: stochasticK            — (close-min14)/(max14-min14) — position within H-L range
//   5: macdHistNorm           — (macdLine-signalLine)/close — histogram instead of line only
//   6: bollingerWidth         — 4*std20/sma20 — breakout signal
//   7: smaDistance            — (close-sma20)/close — continuous distance instead of binary 0/1
//
// Optional extension (requires num_features = 9, inactive):
//   8: atrNorm                — ATR(14)/close — real risk per step

/**
 * V2 indicator pre-computation (inactive).
 * Same interface as precomputeIndicators() — drop-in replacement when ready.
 */
function precomputeIndicatorsV2(candles, normWindow) {
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

    const closesArr  = Array.from(closes);
    const highsArr   = Array.from(highs);
    const lowsArr    = Array.from(lows);
    const volumesArr = Array.from(volumes);

    const mean       = rollingMean(closesArr, normWindow);
    const std        = rollingStd(closesArr, normWindow);
    const { macdLine, signalLine } = calculateMACD(closesArr);
    const sma20      = calculateSMA(closesArr, 20);
    const std20      = rollingStd(closesArr, 20);
    const meanVolume = rollingMean(volumesArr, normWindow);
    const stochK     = calculateStochasticK(highsArr, lowsArr, closesArr);

    const f = [
        new Float64Array(n), // 0: normalizedClose
        new Float64Array(n), // 1: relativeRange
        new Float64Array(n), // 2: candleDirection
        new Float64Array(n), // 3: volumeClipped
        new Float64Array(n), // 4: stochasticK
        new Float64Array(n), // 5: macdHistNorm
        new Float64Array(n), // 6: bollingerWidth
        new Float64Array(n), // 7: smaDistance
    ];

    for (let i = 0; i < n; i++) {
        const c = closes[i], o = opens[i], h = highs[i], l = lows[i], v = volumes[i];
        f[0][i] = std[i]        !== 0 ? (c - mean[i]) / std[i]                : 0;
        f[1][i] = c             !== 0 ? (h - l) / c                           : 0;
        f[2][i] = c             !== 0 ? (c - o) / c                           : 0;
        f[3][i] = meanVolume[i] !== 0 ? Math.min(v / meanVolume[i], 3.0)      : 0;
        f[4][i] = stochK[i];
        f[5][i] = c             !== 0 ? (macdLine[i] - signalLine[i]) / c     : 0;
        f[6][i] = sma20[i]      !== 0 ? (4 * std20[i]) / sma20[i]            : 0;
        f[7][i] = c             !== 0 ? (c - sma20[i]) / c                   : 0;
    }

    return f;
}

/**
 * Assembles v2 feature matrix [numCandles × 8] from pre-computed v2 arrays.
 */
function assembleFeaturesFromPrecomputedV2(f, startIdx, endIdx, numCandles) {
    const actualLen = endIdx - startIdx;
    const padding   = numCandles - actualLen;
    const result    = new Array(numCandles);

    for (let i = 0; i < padding; i++) {
        result[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    for (let i = 0; i < actualLen; i++) {
        const si = startIdx + i;
        result[padding + i] = [f[0][si], f[1][si], f[2][si], f[3][si], f[4][si], f[5][si], f[6][si], f[7][si]];
    }

    return result;
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
    assembleFeaturesFromPrecomputed,
    buildStateFromPrecomputed,
    TIMEFRAME_KEYS,
    TIMEFRAME_CONFIG_KEYS,
    NUM_FEATURES,
    // v2 (inactive)
    precomputeIndicatorsV2,
    assembleFeaturesFromPrecomputedV2,
};
