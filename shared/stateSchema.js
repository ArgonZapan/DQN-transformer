/**
 * Wspólny schemat stanu dla Python i Node.js.
 * Definiuje strukturę state przesyłaną przez ZMQ.
 */

const STATE_TIMEFRAMES = ['1m', '15m', '1h', '1d', '1w'];

const ACTIONS = {
    LONG: 0,
    SHORT: 1,
    HOLD: 2,
    CLOSE: 3
};

const ACTION_NAMES = ['LONG', 'SHORT', 'HOLD', 'CLOSE'];

function validateState(state) {
    if (!state || typeof state !== 'object') return false;

    for (const tf of STATE_TIMEFRAMES) {
        if (!(tf in state)) return false;
        if (!Array.isArray(state[tf])) return false;
    }

    return true;
}

function getActionName(actionId) {
    return ACTION_NAMES[actionId] || 'UNKNOWN';
}

module.exports = { STATE_TIMEFRAMES, ACTIONS, ACTION_NAMES, validateState, getActionName };
