/**
 * System nagradzania — realized P&L z karami za drawdown i zbyt długie trzymanie pozycji.
 * Nagroda końcowa dawana tylko przy zamknięciu pozycji.
 * Wszystkie wartości z config.toml [reward].
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
    // Natychmiastowy koszt wejścia w pozycję — model czuje go zaraz po otwarciu.
    // Jeśli otwarcie następuje zbyt szybko po zamknięciu (cooldown), trade_penalty jest podwojona.
    const cooldown = rewardConfig.post_close_cooldown_steps ?? 0;
    const inCooldown = cooldown > 0 && stepsSinceClose < cooldown;
    const tradePenalty = inCooldown ? rewardConfig.trade_penalty * 2 : rewardConfig.trade_penalty;
    return -(rewardConfig.commission_open + tradePenalty);
}

function calculateReward(position, rewardConfig) {
    // Nagroda przy zamknięciu:
    //   1. PnL zdyskontowany za zbyt długie trzymanie (time decay)
    //   2. Minus prowizja zamknięcia i close_penalty
    //   3. Minus kara za maksymalny drawdown podczas trzymania
    // (prowizja za otwarcie i trade_penalty zostały już odliczone przy otwarciu)
    if (!position || !position.closed) return 0;

    const pnl = calculatePnL(position);

    // Time decay: PnL mnożony przez 1/(1 + t/half_life).
    // Przy t=0 → decay=1.0 (brak kary). Przy t=time_decay_hours → decay=0.5 (połowa PnL).
    // Zniechęca do przetrzymywania pozycji bez wyraźnego trendu.
    const holdingTimeHours = (position.closeTime - position.openTime) / 3_600_000;
    const timeDecay = 1.0 / (1.0 + holdingTimeHours / rewardConfig.time_decay_hours);
    const pnlDecayed = pnl * timeDecay;

    // Kara za drawdown: im głębszy obsunięcie kapitału podczas trzymania, tym większa kara.
    // maxDrawdown jest w [0, 1] (frakcja ceny), drawdown_penalty skaluje jej wpływ.
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
