const assert = require('assert');
const { calculatePnL, calculateReward, calculateRewardRiskAdjusted, calculateRewardTimeDecay } = require('../../node/env/reward');
const { Episode } = require('../../node/env/episode');
const { getActionMask, buildFeatures } = require('../../node/env/state');
const { Position, ACTIONS } = require('../../node/env/tradingEnv');

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

function approxEqual(a, b, tol = 0.0001) {
    return Math.abs(a - b) < tol;
}

const rewardConfig = {
    commission_open: 0.001,
    commission_close: 0.001,
    trade_penalty: 0.001,
    clip_min: -1.0,
    clip_max: 1.0,
    drawdown_penalty: 0.5,
    time_decay_hours: 1.0
};

console.log('Reward Tests\n');

test('PnL LONG positive', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: 110, closed: true };
    assert.ok(approxEqual(calculatePnL(pos), 0.1));
});

test('PnL SHORT positive', () => {
    const pos = { side: 'SHORT', openPrice: 110, closePrice: 100, closed: true };
    assert.ok(approxEqual(calculatePnL(pos), 0.0909, 0.001));
});

test('PnL returns 0 for open position', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: null, closed: false };
    assert.strictEqual(calculatePnL(pos), 0);
});

test('reward includes commission and penalty', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: 110, closed: true };
    const reward = calculateReward(pos, rewardConfig);
    const expected = 0.1 - 0.001 - 0.001 - 0.001;
    assert.ok(approxEqual(reward, expected));
});

test('reward is clipped to [-1, 1]', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: 500, closed: true };
    const reward = calculateReward(pos, rewardConfig);
    assert.strictEqual(reward, 1.0);
});

test('reward is 0 for null position', () => {
    assert.strictEqual(calculateReward(null, rewardConfig), 0);
});

test('risk-adjusted reward penalizes drawdown', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: 110, closed: true };
    const regular = calculateReward(pos, rewardConfig);
    const adjusted = calculateRewardRiskAdjusted(pos, 0.05, rewardConfig);
    assert.ok(adjusted < regular);
});

test('time decay reward decreases with time', () => {
    const pos = { side: 'LONG', openPrice: 100, closePrice: 110, closed: true };
    const fast = calculateRewardTimeDecay(pos, 0.5, rewardConfig);
    const slow = calculateRewardTimeDecay(pos, 10, rewardConfig);
    assert.ok(fast > slow);
});

console.log('\nEpisode Tests\n');

const episodeConfig = {
    training: { train_data_fraction: 0.8, min_episode_length: 100, gamma: 0.99 }
};

test('episode random start is within bounds', () => {
    const ep = new Episode(episodeConfig);
    const dataLen = 1000;
    for (let i = 0; i < 50; i++) {
        const start = ep.getRandomStartIndex(dataLen);
        const trainEnd = Math.floor(dataLen * 0.8);
        assert.ok(start >= 0, `start=${start} < 0`);
        assert.ok(start <= trainEnd - 100, `start=${start} > ${trainEnd - 100}`);
    }
});

test('episode tracks steps', () => {
    const ep = new Episode(episodeConfig);
    ep.start(1000);
    ep.addStep({}, 0, 0.1, {}, false, [1,1,1,0]);
    ep.addStep({}, 2, 0, {}, false, [0,0,1,1]);
    assert.strictEqual(ep.getStepCount(), 2);
});

test('Monte Carlo returns calculated correctly', () => {
    const ep = new Episode({ training: { train_data_fraction: 0.8, min_episode_length: 10, gamma: 0.5 } });
    ep.start(1000);
    ep.addStep({}, 0, 1.0, {}, false, []);
    ep.addStep({}, 0, 2.0, {}, false, []);
    ep.addStep({}, 0, 3.0, {}, true, []);

    const experiences = ep.getExperiences();
    assert.ok(approxEqual(experiences[2].returnG, 3.0));
    assert.ok(approxEqual(experiences[1].returnG, 2.0 + 0.5 * 3.0));
    assert.ok(approxEqual(experiences[0].returnG, 1.0 + 0.5 * (2.0 + 0.5 * 3.0)));
});

test('episode detects train end', () => {
    const ep = new Episode(episodeConfig);
    ep.start(1000);
    ep.currentIndex = 799;
    assert.ok(ep.isAtTrainEnd(1000));
});

console.log('\nState Tests\n');

test('action mask - no position', () => {
    const mask = getActionMask(null);
    assert.deepStrictEqual(mask, [1, 1, 1, 0]);
});

test('action mask - position open', () => {
    const mask = getActionMask({ side: 'LONG' });
    assert.deepStrictEqual(mask, [0, 0, 1, 1]);
});

test('buildFeatures returns 8 features per candle', () => {
    const candles = Array.from({ length: 30 }, (_, i) => ({
        open: 100 + i, high: 102 + i, low: 99 + i, close: 101 + i, volume: 1000 + i * 10
    }));
    const features = buildFeatures(candles, 20);
    assert.strictEqual(features.length, 30);
    assert.strictEqual(features[0].length, 8);
});

console.log('\nPosition Tests\n');

test('Position tracks drawdown for LONG', () => {
    const pos = new Position('LONG', 100, Date.now());
    pos.updateDrawdown(110);
    pos.updateDrawdown(105);
    assert.ok(pos.maxDrawdown > 0);
});

test('Position close sets fields', () => {
    const pos = new Position('SHORT', 100, 1000);
    pos.close(95, 2000);
    assert.strictEqual(pos.closePrice, 95);
    assert.strictEqual(pos.closeTime, 2000);
    assert.strictEqual(pos.closed, true);
});

console.log('\nDone.');
