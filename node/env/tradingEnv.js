const { buildState, getActionMask } = require('./state');
const { calculateOpenPenalty, calculateReward } = require('./reward');
const { Episode } = require('./episode');

const ACTIONS = { LONG: 0, SHORT: 1, HOLD: 2, CLOSE: 3 };

class Position {
    constructor(side, openPrice, openTime) {
        this.side = side;
        this.openPrice = openPrice;
        this.openTime = openTime;
        this.closePrice = null;
        this.closeTime = null;
        this.closed = false;
        this.maxDrawdown = 0;
        this.peakPnl = 0;
    }

    updateDrawdown(currentPrice) {
        let currentPnl;
        if (this.side === 'LONG') {
            currentPnl = (currentPrice - this.openPrice) / this.openPrice;
        } else {
            currentPnl = (this.openPrice - currentPrice) / this.openPrice;
        }

        if (currentPnl > this.peakPnl) this.peakPnl = currentPnl;
        const drawdown = this.peakPnl - currentPnl;
        if (drawdown > this.maxDrawdown) this.maxDrawdown = drawdown;
    }

    close(closePrice, closeTime) {
        this.closePrice = closePrice;
        this.closeTime = closeTime;
        this.closed = true;
    }
}

class TradingEnv {
    constructor(config, symbol) {
        this.config = config;
        this.symbol = symbol;
        this.rewardConfig = config.reward;

        this.stepInterval = config.training.step_interval || 1;
        this.maxTradesPerEpisode = config.training.max_trades_per_episode ?? 1;
        this.position = null;
        this.episode = new Episode(config);
        this.allCandles = {};
        this.currentStepIndex = 0;
        this.trades = [];
    }

    setData(candlesPerTimeframe) {
        this.allCandles = candlesPerTimeframe;
    }

    getDataLength() {
        const tf1m = this.allCandles['1m'];
        return tf1m ? tf1m.length : 0;
    }

    reset() {
        this.position = null;
        this.trades = [];
        const dataLen = this.getDataLength();
        this.currentStepIndex = this.episode.start(dataLen);
        return this._getState();
    }

    _getState() {
        const candle1m = this.allCandles['1m'];
        if (!candle1m || this.currentStepIndex >= candle1m.length) {
            return null;
        }
        const currentTime = candle1m[this.currentStepIndex].close_time || Date.now();
        return buildState(this.allCandles, currentTime, this.config);
    }

    _getCurrentPrice() {
        const candle1m = this.allCandles['1m'];
        if (!candle1m || this.currentStepIndex >= candle1m.length) return 0;
        return candle1m[this.currentStepIndex].close;
    }

    _getCurrentTime() {
        const candle1m = this.allCandles['1m'];
        if (!candle1m || this.currentStepIndex >= candle1m.length) return 0;
        return candle1m[this.currentStepIndex].timestamp;
    }

    step(action) {
        const currentPrice = this._getCurrentPrice();
        const currentTime = this._getCurrentTime();
        let reward = 0;
        let tradeClosed = false;

        if (action === ACTIONS.LONG && this.position === null) {
            this.position = new Position('LONG', currentPrice, currentTime);
            reward = calculateOpenPenalty(this.rewardConfig);
        } else if (action === ACTIONS.SHORT && this.position === null) {
            this.position = new Position('SHORT', currentPrice, currentTime);
            reward = calculateOpenPenalty(this.rewardConfig);
        } else if (action === ACTIONS.CLOSE && this.position !== null) {
            this.position.close(currentPrice, currentTime);
            reward = calculateReward(this.position, this.rewardConfig);
            this.trades.push({ ...this.position });
            this.position = null;
            if (this.maxTradesPerEpisode > 0 && this.trades.length >= this.maxTradesPerEpisode) {
                tradeClosed = true;
            }
        }

        if (this.position !== null) {
            this.position.updateDrawdown(currentPrice);
        }

        // Capture state BEFORE incrementing index (Bug 6: addStep used wrong state)
        const currentState = this._getState();

        this.currentStepIndex += this.stepInterval;

        const done = tradeClosed || this.episode.isAtTrainEnd(this.getDataLength()) || this._getState() === null;

        if (done && this.position !== null) {
            this.position.close(currentPrice, currentTime);
            reward = calculateReward(this.position, this.rewardConfig);
            this.trades.push({ ...this.position });
            this.position = null;
        }

        const nextState = done ? null : this._getState();
        const actionMask = getActionMask(this.position);

        this.episode.addStep(
            currentState,
            action,
            reward,
            nextState,
            done,
            actionMask
        );

        return { nextState, reward, done, actionMask };
    }

    getActionMask() {
        return getActionMask(this.position);
    }

    getCurrentPrice() {
        return this._getCurrentPrice();
    }

    getCurrentTimestamp() {
        return this._getCurrentTime();
    }

    getEpisodeExperiences() {
        return this.episode.getExperiences();
    }

    getTrades() {
        return this.trades;
    }

    getMetrics() {
        const trades = this.trades;
        if (trades.length === 0) {
            return { win_rate: 0, profit_factor: 0, total_trades: 0 };
        }

        let wins = 0, losses = 0;
        let totalProfit = 0, totalLoss = 0;

        const commission = this.rewardConfig.commission_open + this.rewardConfig.commission_close;
        const tradePenalty = this.rewardConfig.trade_penalty;
        for (const trade of trades) {
            const gross = trade.side === 'SHORT'
                ? (trade.openPrice - trade.closePrice) / trade.openPrice
                : (trade.closePrice - trade.openPrice) / trade.openPrice;
            const pnl = gross - commission - tradePenalty;
            if (pnl > 0) { wins++; totalProfit += pnl; }
            else { losses++; totalLoss += Math.abs(pnl); }
        }

        const winRate = trades.length > 0 ? wins / trades.length : 0;
        const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;

        return {
            win_rate: winRate,
            profit_factor: profitFactor,
            total_trades: trades.length,
            wins,
            losses
        };
    }
}

module.exports = { TradingEnv, Position, ACTIONS };
