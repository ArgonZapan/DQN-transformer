/**
 * Reward system — realized P&L with penalties for drawdown and holding too long.
 * Final reward given only on position close.
 * All values from config.toml [reward].
 */

function calculatePnL(position) {
    if (!position || !position.closed) return 0;

    if (position.side === 'LONG') {
        return (position.closePrice - position.openPrice) / position.openPrice;
    } else if (position.side === 'SHORT') {
        return (position.openPrice - position.closePrice) / position.openPrice;
    }

    return 0;
}

function calculateOpenPenalty(rewardConfig, stepsSinceClose = Infinity) {
    // Immediate cost of entering a position — model feels it right after opening.
    // If opening occurs too soon after a close (cooldown), trade_penalty is doubled.
    const cooldown = rewardConfig.post_close_cooldown_steps ?? 0;
    const inCooldown = cooldown > 0 && stepsSinceClose < cooldown;
    const tradePenalty = inCooldown ? rewardConfig.trade_penalty * 2 : rewardConfig.trade_penalty;
    return -(rewardConfig.commission_open + tradePenalty);
}

function calculateReward(position, rewardConfig) {
    // Reward on close:
    //   1. PnL discounted for holding too long (time decay)
    //   2. Minus close commission and close_penalty
    //   3. Minus penalty for max drawdown during holding
    // (open commission and trade_penalty were already deducted at open)
    if (!position || !position.closed) return 0;

    const pnl = calculatePnL(position);

    // Time decay: PnL multiplied by 1/(1 + t/half_life).
    // At t=0 → decay=1.0 (no penalty). At t=time_decay_hours → decay=0.5 (half PnL).
    // Discourages holding positions without a clear trend.
    const holdingTimeHours = (position.closeTime - position.openTime) / 3_600_000;
    const timeDecay = 1.0 / (1.0 + holdingTimeHours / rewardConfig.time_decay_hours);
    const pnlDecayed = pnl * timeDecay;

    // Drawdown penalty: deeper drawdown during holding → larger penalty.
    // maxDrawdown is in [0, 1] (price fraction), drawdown_penalty scales its impact.
    const drawdownPenalty = (position.maxDrawdown || 0) * rewardConfig.drawdown_penalty;

    const closePenalty = rewardConfig.close_penalty ?? 0;
    let reward = pnlDecayed - rewardConfig.commission_close - closePenalty - drawdownPenalty;
    reward = Math.max(rewardConfig.clip_min, Math.min(rewardConfig.clip_max, reward));

    return reward;
}

function calculatePnLDecayed(position, rewardConfig) {
    if (!position || !position.closed) return 0;
    const pnl = calculatePnL(position);
    const holdingTimeHours = (position.closeTime - position.openTime) / 3_600_000;
    const timeDecay = 1.0 / (1.0 + holdingTimeHours / rewardConfig.time_decay_hours);
    return pnl * timeDecay;
}

module.exports = {
    calculatePnL,
    calculatePnLDecayed,
    calculateOpenPenalty,
    calculateReward,
};
