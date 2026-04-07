const { calculateRSI, calculateMACD, calculateSMA, rollingMean, rollingStd } = require('../data/indicators');

const TIMEFRAME_KEYS = ['1m', '15m', '1h', '1d', '1w'];
const TIMEFRAME_CONFIG_KEYS = {
    '1m': 'candles_1m',
    '15m': 'candles_15m',
    '1h': 'candles_1h',
    '1d': 'candles_1d',
    '1w': 'candles_1w'
};

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
 * aby uniknąć look-ahead bias
 */
function getAlignedCandles(allCandles, currentTime, timeframe, numCandles) {
    const filtered = allCandles.filter(c => c.close_time <= currentTime);
    return filtered.slice(-numCandles);
}

function buildState(allCandlesPerTf, currentTime, config) {
    const timeframesConfig = config.timeframes;
    const normWindow = config.data.normalization_window;
    const state = {};

    for (const tf of TIMEFRAME_KEYS) {
        const configKey = TIMEFRAME_CONFIG_KEYS[tf];
        const numCandles = timeframesConfig[configKey];
        const candles = allCandlesPerTf[tf] || [];

        const aligned = getAlignedCandles(candles, currentTime, tf, numCandles);

        if (aligned.length < numCandles) {
            const features = buildFeatures(aligned, normWindow);
            const padding = numCandles - aligned.length;
            const paddedFeatures = [];
            for (let i = 0; i < padding; i++) {
                paddedFeatures.push([0, 0, 0, 0, 0, 0, 0, 0]);
            }
            state[tf] = paddedFeatures.concat(features);
        } else {
            state[tf] = buildFeatures(aligned, normWindow);
        }
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
    TIMEFRAME_KEYS,
    TIMEFRAME_CONFIG_KEYS
};
