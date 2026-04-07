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
    getFeaturesConfig
} = require('../../node/config');

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
    assert.ok(config);
    assert.ok(typeof config === 'object');
});

test('throws on missing file', () => {
    assert.throws(() => loadConfig('/nonexistent/config.toml'), /not found/);
});

// Learner config
test('learner config has correct values', () => {
    const learner = getLearnerConfig(config);
    assert.strictEqual(learner.host, 'tcp://127.0.0.1');
    assert.strictEqual(learner.port, 5555);
    assert.strictEqual(learner.metrics_port, 5556);
    assert.strictEqual(learner.device, 'cuda');
});

// Actors config
test('actors config has 3 entries', () => {
    const actors = getActorsConfig(config);
    assert.strictEqual(actors.length, 3);
});

test('actors have correct symbols', () => {
    const actors = getActorsConfig(config);
    const symbols = actors.map(a => a.symbol);
    assert.ok(symbols.includes('BTCUSDT'));
    assert.ok(symbols.includes('ETHUSDT'));
    assert.ok(symbols.includes('SOLUSDT'));
});

// Timeframes config
test('timeframes config has correct values', () => {
    const tf = getTimeframesConfig(config);
    assert.strictEqual(tf.candles_1m, 15);
    assert.strictEqual(tf.candles_15m, 15);
    assert.strictEqual(tf.candles_1h, 20);
    assert.strictEqual(tf.candles_1d, 30);
    assert.strictEqual(tf.candles_1w, 54);
});

// Training config
test('training config has correct values', () => {
    const training = getTrainingConfig(config);
    assert.strictEqual(training.gamma, 0.999);
    assert.strictEqual(training.lr, 0.0001);
    assert.strictEqual(training.batch_size, 256);
    assert.strictEqual(training.buffer_capacity, 500000);
    assert.strictEqual(training.epsilon_start, 1.0);
    assert.strictEqual(training.epsilon_end, 0.05);
    assert.strictEqual(training.seed, 42);
});

// Monitoring config
test('monitoring config has correct values', () => {
    const mon = getMonitoringConfig(config);
    assert.strictEqual(mon.port, 3001);
    assert.strictEqual(mon.metrics_pull_port, 3002);
    assert.strictEqual(mon.metrics_push_interval_sec, 5);
    assert.strictEqual(mon.dashboard_poll_interval_sec, 60);
});

// Data config
test('data config has correct values', () => {
    const data = getDataConfig(config);
    assert.strictEqual(data.source, 'file');
    assert.strictEqual(data.normalization_window, 20);
    assert.strictEqual(data.binance_rate_limit, 1000);
});

// Reward config
test('reward config has correct values', () => {
    const reward = getRewardConfig(config);
    assert.strictEqual(reward.commission_open, 0.001);
    assert.strictEqual(reward.commission_close, 0.001);
    assert.strictEqual(reward.clip_min, -1.0);
    assert.strictEqual(reward.clip_max, 1.0);
});

// Features config
test('features config has correct values', () => {
    const features = getFeaturesConfig(config);
    assert.strictEqual(features.num_features, 8);
    assert.strictEqual(features.use_ohlcv, true);
    assert.strictEqual(features.use_rsi, true);
});

console.log('\nDone.');
