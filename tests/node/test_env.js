const assert = require('assert');
const path = require('path');
const {
    calculatePnL, calculatePnLDecayed, calculateOpenPenalty, calculateReward,
} = require(path.join(__dirname, '..', '..', 'node', 'env', 'reward'));
const { Episode } = require(path.join(__dirname, '..', '..', 'node', 'env', 'episode'));
const { getActionMask, buildFeatures, NUM_FEATURES } = require(path.join(__dirname, '..', '..', 'node', 'env', 'state'));
const { Position, ACTIONS } = require(path.join(__dirname, '..', '..', 'node', 'env', 'tradingEnv'));

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
    time_decay_hours: 1.0,
    close_penalty: 0,
    post_close_cooldown_steps: 0,
};

// Build a closed position with explicit hold time. holdHours=0 → time decay is a no-op.
function closedPos(side, openPrice, closePrice, holdHours = 0) {
    return {
        side, openPrice, closePrice, closed: true,
        openTime: 0, closeTime: holdHours * 3_600_000,
        maxDrawdown: 0, peakPnl: 0,
    };
}

console.log('Reward Tests\n');

test('PnL LONG positive', () => {
    assert.ok(approxEqual(calculatePnL(closedPos('LONG', 100, 110)), 0.1));
});

test('PnL SHORT positive', () => {
    assert.ok(approxEqual(calculatePnL(closedPos('SHORT', 110, 100)), 0.0909, 0.001));
});

test('PnL returns 0 for open position', () => {
    assert.strictEqual(calculatePnL({ side: 'LONG', openPrice: 100, closePrice: null, closed: false }), 0);
});

test('reward on close = pnl - close commission (open fee already charged at open)', () => {
    // 0 hold hours → no time decay; no drawdown → no drawdown penalty.
    const reward = calculateReward(closedPos('LONG', 100, 110, 0), rewardConfig);
    const expected = 0.1 - rewardConfig.commission_close;
    assert.ok(approxEqual(reward, expected), `reward=${reward} expected=${expected}`);
});

test('reward is clipped to clip_max', () => {
    assert.strictEqual(calculateReward(closedPos('LONG', 100, 500, 0), rewardConfig), 1.0);
});

test('reward is 0 for null position', () => {
    assert.strictEqual(calculateReward(null, rewardConfig), 0);
});

test('time decay reduces realized PnL with longer holds', () => {
    const fast = calculatePnLDecayed(closedPos('LONG', 100, 110, 0.5), rewardConfig);
    const slow = calculatePnLDecayed(closedPos('LONG', 100, 110, 10), rewardConfig);
    assert.ok(fast > slow, `fast=${fast} should exceed slow=${slow}`);
});

test('open penalty charges commission + trade penalty', () => {
    const p = calculateOpenPenalty(rewardConfig, Infinity);
    assert.ok(approxEqual(p, -(rewardConfig.commission_open + rewardConfig.trade_penalty)));
});

test('open penalty doubles trade penalty within post-close cooldown', () => {
    const cfg = { ...rewardConfig, post_close_cooldown_steps: 10 };
    const inCooldown  = calculateOpenPenalty(cfg, 5);
    const outCooldown = calculateOpenPenalty(cfg, 50);
    assert.ok(inCooldown < outCooldown, `inCooldown=${inCooldown} should be < outCooldown=${outCooldown}`);
});

console.log('\nEpisode Tests\n');

const episodeConfig = {
    training: { validation_days: 0, episode_length: 100, step_interval: 1, gamma: 0.99 },
    timeframes: { candles_1d: 0 },
};

test('episode random start is within bounds', () => {
    const ep = new Episode(episodeConfig);
    const dataLen = 1000;
    for (let i = 0; i < 50; i++) {
        const start = ep.getRandomStartIndex(dataLen);
        assert.ok(start >= 0, `start=${start} < 0`);
        assert.ok(start <= dataLen - 100, `start=${start} > ${dataLen - 100}`);
    }
});

test('episode tracks steps', () => {
    const ep = new Episode(episodeConfig);
    ep.start(1000);
    ep.addStep({}, 0, 0.1, {}, false, [1, 1, 1, 0]);
    ep.addStep({}, 2, 0, {}, false, [0, 0, 1, 1]);
    assert.strictEqual(ep.getStepCount(), 2);
});

test('Monte Carlo returns calculated correctly', () => {
    const ep = new Episode({
        training: { validation_days: 0, episode_length: 10, step_interval: 1, gamma: 0.5 },
        timeframes: { candles_1d: 0 },
    });
    ep.start(1000);
    ep.addStep({}, 0, 1.0, {}, false, []);
    ep.addStep({}, 0, 2.0, {}, false, []);
    ep.addStep({}, 0, 3.0, {}, true, []);

    const experiences = ep.getExperiences();
    assert.ok(approxEqual(experiences[2].returnG, 3.0));
    assert.ok(approxEqual(experiences[1].returnG, 2.0 + 0.5 * 3.0));
    assert.ok(approxEqual(experiences[0].returnG, 1.0 + 0.5 * (2.0 + 0.5 * 3.0)));
});

test('n-step experiences carry discounted reward, gammaToN and the real nextState', () => {
    const ep = new Episode({
        training: { validation_days: 0, episode_length: 10, step_interval: 1, gamma: 0.9 },
        timeframes: { candles_1d: 0 },
    });
    ep.start(100);
    for (let i = 0; i < 5; i++) ep.addStep({ id: i }, 2, 1.0, {}, i === 4, [1, 1, 1, 0]);
    const exps = ep.getExperiencesNStep(2);
    assert.strictEqual(exps.length, 5);
    assert.ok(approxEqual(exps[0].reward, 1.0 + 0.9 * 1.0), `reward=${exps[0].reward}`);
    assert.ok(approxEqual(exps[0].gammaToN, 0.81), `gammaToN=${exps[0].gammaToN}`);
    assert.deepStrictEqual(exps[0].nextState, { id: 2 }); // nextState = steps[t+n].state
    assert.strictEqual(exps[4].done, true);              // last step terminates within n
    assert.strictEqual(exps[4].nextState, null);
});

test('episode detects train end', () => {
    const ep = new Episode(episodeConfig);
    ep.start(1000);
    ep.startIndex = 0;
    ep.currentIndex = 99; // epEnd = 0 + 100*1 = 100; train end at currentIndex >= 99
    assert.ok(ep.isAtTrainEnd(1000));
    ep.currentIndex = 50;
    assert.ok(!ep.isAtTrainEnd(1000));
});

console.log('\nState Tests\n');

test('action mask - no position', () => {
    assert.deepStrictEqual(getActionMask(null), [1, 1, 1, 0]);
});

test('action mask - position open', () => {
    assert.deepStrictEqual(getActionMask({ side: 'LONG' }), [0, 0, 1, 1]);
});

test(`buildFeatures returns ${NUM_FEATURES} features per candle`, () => {
    const candles = Array.from({ length: 30 }, (_, i) => ({
        open: 100 + i, high: 102 + i, low: 99 + i, close: 101 + i, volume: 1000 + i * 10,
    }));
    const features = buildFeatures(candles, 20);
    assert.strictEqual(features.length, 30);
    assert.strictEqual(features[0].length, NUM_FEATURES);
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
