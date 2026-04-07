const assert = require('assert');
const { calculateRSI, calculateEMA, calculateMACD, calculateSMA, rollingMean, rollingStd } = require('../../node/data/indicators');
const { PerPairNormalizer } = require('../../node/data/normalizer');

function test(name, fn) {
    try {
        fn();
        console.log(`  ✓ ${name}`);
    } catch (err) {
        console.log(`  ✗ ${name}`);
        console.log(`    ${err.message}`);
        process.exitCode = 1;
    }
}

function approxEqual(a, b, tolerance = 0.01) {
    return Math.abs(a - b) < tolerance;
}

console.log('Indicators Tests\n');

// RSI tests
test('RSI returns array of same length', () => {
    const closes = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64];
    const rsi = calculateRSI(closes);
    assert.strictEqual(rsi.length, closes.length);
});

test('RSI values are in [0, 100]', () => {
    const closes = Array.from({length: 30}, (_, i) => 100 + Math.sin(i * 0.5) * 10);
    const rsi = calculateRSI(closes);
    for (const val of rsi) {
        assert.ok(val >= 0 && val <= 100, `RSI ${val} out of range`);
    }
});

test('RSI is 0 for first elements (before period)', () => {
    const closes = Array.from({length: 30}, (_, i) => 100 + i);
    const rsi = calculateRSI(closes);
    for (let i = 0; i < 14; i++) {
        assert.strictEqual(rsi[i], 0);
    }
});

// EMA tests
test('EMA returns array of same length', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const ema = calculateEMA(values, 3);
    assert.strictEqual(ema.length, values.length);
});

test('EMA first value equals input', () => {
    const values = [10, 20, 30, 40, 50];
    const ema = calculateEMA(values, 3);
    assert.strictEqual(ema[0], 10);
});

// MACD tests
test('MACD returns three arrays', () => {
    const closes = Array.from({length: 50}, (_, i) => 100 + Math.sin(i * 0.3) * 5);
    const { macdLine, signalLine, histogram } = calculateMACD(closes);
    assert.strictEqual(macdLine.length, closes.length);
    assert.strictEqual(signalLine.length, closes.length);
    assert.strictEqual(histogram.length, closes.length);
});

// SMA tests
test('SMA with period=1 equals input', () => {
    const values = [10, 20, 30, 40];
    const sma = calculateSMA(values, 1);
    for (let i = 0; i < values.length; i++) {
        assert.ok(approxEqual(sma[i], values[i]), `SMA[${i}]=${sma[i]} != ${values[i]}`);
    }
});

test('SMA is correct for known values', () => {
    const values = [2, 4, 6, 8, 10];
    const sma = calculateSMA(values, 3);
    assert.ok(approxEqual(sma[2], 4));
    assert.ok(approxEqual(sma[3], 6));
    assert.ok(approxEqual(sma[4], 8));
});

// Rolling stats
test('rollingMean equals SMA', () => {
    const values = [1, 2, 3, 4, 5];
    const mean = rollingMean(values, 3);
    const sma = calculateSMA(values, 3);
    for (let i = 0; i < values.length; i++) {
        assert.ok(approxEqual(mean[i], sma[i]));
    }
});

test('rollingStd returns correct values', () => {
    const values = [10, 10, 10, 10];
    const std = rollingStd(values, 3);
    assert.ok(approxEqual(std[3], 0));
});

console.log('\nNormalizer Tests\n');

// Normalizer tests
test('PerPairNormalizer cold start returns 0', () => {
    const norm = new PerPairNormalizer(20);
    assert.strictEqual(norm.normalize(100), 0);
});

test('PerPairNormalizer with single value returns 0', () => {
    const norm = new PerPairNormalizer(20);
    norm.update(100);
    assert.strictEqual(norm.normalize(100), 0);
});

test('PerPairNormalizer normalizes correctly', () => {
    const norm = new PerPairNormalizer(5);
    [100, 102, 98, 101, 99].forEach(v => norm.update(v));
    const result = norm.normalize(100);
    assert.ok(typeof result === 'number');
    assert.ok(!isNaN(result));
});

test('PerPairNormalizer window limit', () => {
    const norm = new PerPairNormalizer(3);
    [1, 2, 3, 4, 5].forEach(v => norm.update(v));
    assert.strictEqual(norm.history.length, 3);
    assert.deepStrictEqual(norm.history, [3, 4, 5]);
});

test('PerPairNormalizer save/load state', () => {
    const norm1 = new PerPairNormalizer(5);
    [100, 102, 98].forEach(v => norm1.update(v));
    const state = norm1.getState();

    const norm2 = new PerPairNormalizer(1);
    norm2.loadState(state);
    assert.strictEqual(norm2.window, 5);
    assert.deepStrictEqual(norm2.history, [100, 102, 98]);
    assert.strictEqual(norm1.normalize(100), norm2.normalize(100));
});

console.log('\nDone.');
