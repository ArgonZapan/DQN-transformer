/**
 * System nagradzania — realized P&L only.
 * Nagroda dawana tylko przy zamknięciu pozycji.
 * Wszystkie wartości z config.toml.
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

function calculateOpenPenalty(rewardConfig) {
    // Natychmiastowy koszt wejścia w pozycję — model czuje go zaraz po otwarciu
    return -(rewardConfig.commission_open + rewardConfig.trade_penalty);
}

function calculateReward(position, rewardConfig) {
    // Nagroda przy zamknięciu: tylko PnL i prowizja za zamknięcie
    // (prowizja za otwarcie i trade_penalty zostały już odliczone przy otwarciu)
    if (!position || !position.closed) return 0;

    const pnl = calculatePnL(position);
    let reward = pnl - rewardConfig.commission_close;
    reward = Math.max(rewardConfig.clip_min, Math.min(rewardConfig.clip_max, reward));

    return reward;
}

function calculateRewardRiskAdjusted(position, maxDrawdown, rewardConfig) {
    if (!position || !position.closed) return 0;

    const pnl = calculatePnL(position);
    const commission = rewardConfig.commission_open + rewardConfig.commission_close;
    const tradePenalty = rewardConfig.trade_penalty;
    const riskPenalty = Math.abs(maxDrawdown) * rewardConfig.drawdown_penalty;

    let reward = pnl - commission - tradePenalty - riskPenalty;
    reward = Math.max(rewardConfig.clip_min, Math.min(rewardConfig.clip_max, reward));

    return reward;
}

function calculateRewardTimeDecay(position, holdingTimeHours, rewardConfig) {
    if (!position || !position.closed) return 0;

    const pnl = calculatePnL(position);
    const timeDecay = 1.0 / (1.0 + holdingTimeHours / rewardConfig.time_decay_hours);
    const commission = rewardConfig.commission_open + rewardConfig.commission_close;
    const tradePenalty = rewardConfig.trade_penalty;

    let reward = pnl * timeDecay - commission - tradePenalty;
    reward = Math.max(rewardConfig.clip_min, Math.min(rewardConfig.clip_max, reward));

    return reward;
}

module.exports = {
    calculatePnL,
    calculateOpenPenalty,
    calculateReward,
    calculateRewardRiskAdjusted,
    calculateRewardTimeDecay
};
