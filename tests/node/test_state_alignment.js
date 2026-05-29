/**
 * Regression test: the experience stored for step t must hold the state the agent
 * actually acted on (position features BEFORE the action), not the post-action state.
 *
 * Before the fix, TradingEnv.step() captured currentState after mutating the
 * position, so an OPEN was stored with an already-in-position state and a CLOSE
 * with an already-flat one — desynchronizing (state, action). Run directly:
 *   node tests/node/test_state_alignment.js
 */

const assert = require('assert');
const path = require('path');
const { TradingEnv, ACTIONS } = require(path.join(__dirname, '..', '..', 'node', 'env', 'tradingEnv'));

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

function makeConfig() {
    return {
        reward: {
            commission_open: 0.00075, commission_close: 0.00075, trade_penalty: 0,
            post_close_cooldown_steps: 0, intermediate_reward_max: 0.1, loss_scale: 1.4,
            hold_penalty_per_bar: 0, idle_penalty_per_bar: 0, unrealized_reward_scale: 0.02,
            clip_min: -1, clip_max: 1, reward_scale: 1, time_decay_hours: 4,
            drawdown_penalty: 0, close_penalty: 0,
        },
        training: {
            step_interval: 1, max_trades_per_episode: 1, min_hold_steps: 0,
            validation_days: 0, episode_length: 50, gamma: 0.97,
        },
        timeframes: { candles_1m: 10, candles_15m: 0, candles_1h: 0, candles_1d: 0, candles_1w: 0 },
        data: { normalization_window: 5 },
    };
}

function makeCandles(n = 200) {
    const base = 1700000000000;
    const out = [];
    for (let i = 0; i < n; i++) {
        const p = 100 + Math.sin(i / 5) * 3 + i * 0.05;
        out.push({ open: p, high: p + 1, low: p - 1, close: p, volume: 10 + (i % 7),
                   close_time: base + i * 60000, timestamp: base + i * 60000 });
    }
    return out;
}

function newEnv() {
    const env = new TradingEnv(makeConfig(), 'TESTUSDT');
    env.setData({ '1m': makeCandles() });
    return env;
}

console.log('State/Action Alignment Tests\n');

test('OPEN stores the pre-action (flat) state the agent observed', () => {
    const env = newEnv();
    const s0 = env.reset();
    assert.strictEqual(s0.position[0], 0, 'reset state must be flat (is_long=0)');
    env.step(ACTIONS.LONG);
    const stored = env.episode.steps[0].state;
    assert.strictEqual(stored.position[0], 0, 'stored is_long must be 0 (pre-action), not 1');
    assert.strictEqual(stored.position[1], 0, 'stored is_short must be 0 (pre-action)');
    assert.strictEqual(JSON.stringify(stored.position), JSON.stringify(s0.position),
        'stored state must equal the state the agent acted on');
});

test('CLOSE stores the pre-action (in-position) state the agent observed', () => {
    const env = newEnv();
    let state = env.reset();
    env.step(ACTIONS.LONG);          // open
    state = env.episode.steps[0].nextState;  // what the agent sees before CLOSE (in-position)
    env.step(ACTIONS.CLOSE);         // close
    const stored = env.episode.steps[1].state;
    assert.ok(stored.position[0] === 1 || stored.position[1] === 1,
        'stored state for the CLOSE step must still show an open position (pre-action)');
    assert.strictEqual(JSON.stringify(stored.position), JSON.stringify(state.position),
        'stored CLOSE-step state must equal the in-position state the agent acted on');
});

test('HOLD step state is consistent (no position change)', () => {
    const env = newEnv();
    const s0 = env.reset();
    env.step(ACTIONS.HOLD);
    const stored = env.episode.steps[0].state;
    assert.strictEqual(stored.position[0], 0);
    assert.strictEqual(stored.position[1], 0);
    assert.strictEqual(JSON.stringify(stored.position), JSON.stringify(s0.position));
});

console.log('\nDone.');
