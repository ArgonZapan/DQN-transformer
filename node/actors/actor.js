const { TradingEnv, ACTIONS } = require('../env/tradingEnv');
const { BinanceClient } = require('../data/binance');
const { PerPairNormalizer } = require('../data/normalizer');

const TIMEFRAME_INTERVALS = ['1m', '15m', '1h', '1d', '1w'];

class Actor {
    constructor(actorConfig, globalConfig, pythonClient, monitorClient) {
        this.symbol = actorConfig.symbol;
        this.config = globalConfig;
        this.pythonClient = pythonClient;
        this.monitorClient = monitorClient;

        this.env = new TradingEnv(globalConfig, this.symbol);
        this.binanceClient = new BinanceClient(globalConfig);
        this.normalizer = new PerPairNormalizer(globalConfig.data.normalization_window);

        this.epsilon = globalConfig.training.epsilon_start;
        this.totalSteps = 0;
        this.totalEpisodes = 0;
        this.running = false;
    }

    async loadData() {
        const candlesPerTf = {};
        for (const tf of TIMEFRAME_INTERVALS) {
            try {
                const configKey = `candles_${tf}`;
                const limit = this.config.timeframes[configKey] * 5;
                const candles = await this.binanceClient.getData(this.symbol, tf, limit);
                candlesPerTf[tf] = candles;
                console.log(`[Actor:${this.symbol}] Loaded ${candles.length} candles for ${tf}`);
            } catch (err) {
                if (this.config.data.allow_partial_history) {
                    console.warn(`[Actor:${this.symbol}] Missing data for ${tf}: ${err.message}. Using empty.`);
                    candlesPerTf[tf] = [];
                } else {
                    throw err;
                }
            }
        }
        this.env.setData(candlesPerTf);
    }

    async start() {
        this.running = true;
        console.log(`[Actor:${this.symbol}] Started`);

        while (this.running) {
            try {
                await this.runEpisode();
                this.totalEpisodes++;
            } catch (err) {
                console.error(`[Actor:${this.symbol}] Episode error: ${err.message}`);
                await new Promise(r => setTimeout(r, 1000));
            }
        }
    }

    async runEpisode() {
        let state = this.env.reset();
        if (!state) return;

        let done = false;
        let actionMask = this.env.getActionMask();

        while (!done && this.running) {
            let action;

            if (Math.random() < this.epsilon) {
                const validActions = actionMask
                    .map((m, i) => m === 1 ? i : -1)
                    .filter(i => i >= 0);
                action = validActions[Math.floor(Math.random() * validActions.length)];
            } else {
                try {
                    const response = await this.pythonClient.sendStep({
                        state,
                        action: null,
                        reward: 0,
                        nextState: null,
                        done: false,
                        actionMask
                    });
                    action = response.nextAction;
                } catch (err) {
                    const validActions = actionMask.map((m, i) => m === 1 ? i : -1).filter(i => i >= 0);
                    action = validActions[Math.floor(Math.random() * validActions.length)];
                }
            }

            const result = this.env.step(action);
            done = result.done;
            state = result.nextState;
            actionMask = result.actionMask;

            if (result.reward !== 0 || done) {
                try {
                    await this.pythonClient.sendStep({
                        state: state || {},
                        action,
                        reward: result.reward,
                        nextState: result.nextState || {},
                        done,
                        actionMask
                    });
                } catch (err) {
                    // continue even if send fails
                }
            }

            this.totalSteps++;
        }

        if (this.monitorClient) {
            try {
                await this.monitorClient.sendMetrics(`actor:${this.symbol}`, {
                    epsilon: this.epsilon,
                    totalSteps: this.totalSteps,
                    totalEpisodes: this.totalEpisodes,
                    trades: this.env.getTrades().length
                });
            } catch (err) {
                // non-blocking
            }
        }
    }

    stop() {
        this.running = false;
        console.log(`[Actor:${this.symbol}] Stopped`);
    }
}

module.exports = { Actor };
