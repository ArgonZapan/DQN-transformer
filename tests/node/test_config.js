const assert = require('assert');
const path = require('path');
const {
    loadConfig,
    getActorsConfig,
    getTimeframesConfig,
    getLearnerConfig,
    getTrainingConfig,
    getMonitoringConfig,
    getDataConfig,
    getRewardConfig,
    getFeaturesConfig,
} = require(path.join(__dirname, '..', '..', 'node', 'config'));

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

console.log('Config Loader Tests (Node.js)\n');

const configPath = path.join(__dirname, '..', '..', 'config.toml');
let config;

test('loads valid config', () => {
    config = loadConfig(configPath);
    assert.ok(config && typeof config === 'object');
});

test('throws on missing file', () => {
    assert.throws(() => loadConfig('/nonexistent/config.toml'), /not found/);
});

// Getters return the matching sections (API contract, not brittle values that drift).
test('learner getter exposes connection fields', () => {
    const learner = getLearnerConfig(config);
    assert.strictEqual(learner, config.learner);
    for (const k of ['host', 'port', 'metrics_port', 'device']) {
        assert.ok(k in learner, `learner missing ${k}`);
    }
    assert.ok(typeof learner.device === 'string' && learner.device.length > 0);
});

test('actors getter returns at least one actor with a symbol', () => {
    const actors = getActorsConfig(config);
    assert.ok(Array.isArray(actors) && actors.length >= 1);
    assert.ok(actors.every(a => typeof a.symbol === 'string' && a.symbol.length > 0));
    assert.ok(actors.some(a => a.symbol === 'BTCUSDT'));
});

test('timeframes getter has all required keys and a positive base TF', () => {
    const tf = getTimeframesConfig(config);
    for (const k of ['candles_1m', 'candles_15m', 'candles_1h', 'candles_1d', 'candles_1w']) {
        assert.ok(k in tf, `timeframes missing ${k}`);
        assert.ok(tf[k] >= 0, `${k} must be >= 0`);
    }
    assert.ok(Object.values(tf).some(v => v > 0), 'at least one timeframe must be enabled');
});

test('training getter holds sane invariants', () => {
    const t = getTrainingConfig(config);
    assert.ok(t.gamma > 0 && t.gamma <= 1, `gamma=${t.gamma}`);
    assert.ok(t.lr > 0, `lr=${t.lr}`);
    assert.ok(t.batch_size > 0);
    assert.ok(t.buffer_capacity >= t.min_buffer_size, 'buffer_capacity must be >= min_buffer_size');
    assert.ok(t.epsilon_start >= t.epsilon_end, 'epsilon_start must be >= epsilon_end');
});

test('monitoring getter exposes ports', () => {
    const mon = getMonitoringConfig(config);
    assert.ok(Number.isInteger(mon.port));
    assert.ok(Number.isInteger(mon.metrics_pull_port));
});

test('data getter has source and positive normalization window', () => {
    const data = getDataConfig(config);
    assert.ok(typeof data.source === 'string');
    assert.ok(data.normalization_window > 0);
});

test('reward getter has commissions and a valid clip range', () => {
    const reward = getRewardConfig(config);
    assert.ok(reward.commission_open >= 0 && reward.commission_close >= 0);
    assert.ok(reward.clip_min < reward.clip_max, 'clip_min must be < clip_max');
});

test('features getter reports a positive feature count', () => {
    const features = getFeaturesConfig(config);
    assert.ok(features.num_features > 0);
});

console.log('\nDone.');
