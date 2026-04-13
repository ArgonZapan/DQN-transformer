const Table = require('cli-table3');
const { TradingEnv } = require('../env/tradingEnv');
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
        this.episodeEquity = 0;
        this.cumulativePnl = 0;
    }

    async loadData() {
        const candlesPerTf = {};
        for (const tf of TIMEFRAME_INTERVALS) {
            const configKey   = `candles_${tf}`;
            const stateWindow = this.config.timeframes[configKey] || 0;
            if (stateWindow === 0) continue;  // TF wyłączony w konfiguracji

            try {
                // W trybie file getData zwraca wszystkie dostępne świece (limit ignorowany).
                // W trybie API stateWindow służy jako limit (tylko ostatnie N świec real-time).
                const candles = await this.binanceClient.getData(this.symbol, tf, stateWindow);
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
                const episodeStart = Date.now();
                await this.runEpisode();
                this.totalEpisodes++;
                const episodeTime = Date.now() - episodeStart;
                console.log(`[Actor:${this.symbol}] Episode ${this.totalEpisodes} finished in ${episodeTime}ms, totalSteps=${this.totalSteps}, trades=${this.env.getTrades().length}`);
            } catch (err) {
                console.error(`[Actor:${this.symbol}] Episode error: ${err.message}`);
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        console.log(`[Actor:${this.symbol}] Stopped. Total: ${this.totalEpisodes} episodes, ${this.totalSteps} steps`);
    }

    async runEpisode() {
        let state = this.env.reset();
        if (!state) return;

        let done = false;
        let actionMask = this.env.getActionMask();
        const episodeLog = [];

        // Rozgraj cały epizod zbierając kroki w env.episode
        while (!done && this.running) {
            let action;
            let wasRandom = false;
            let qValues = null;

            // Zapisz cenę i czas przed krokiem (po kroku indeks jest przesunięty)
            const stepPrice = this.env.getCurrentPrice();
            const stepTimestamp = this.env.getCurrentTimestamp();
            const positionBefore = this.env.position ? this.env.position.side : null;
            const maskBeforeStep = [...actionMask];  // maska użyta do decyzji (przed step)

            if (Math.random() < this.epsilon) {
                wasRandom = true;
                // LONG=13%, SHORT=13%, HOLD=60%, CLOSE=13% — ważona losowość
                // Zablokowane akcje (maska=0) dostają wagę 0, reszta renormalizowana
                const WEIGHTS = [13, 13, 60, 13];
                const weights = actionMask.map((m, i) => m === 1 ? WEIGHTS[i] : 0);
                const total = weights.reduce((a, b) => a + b, 0);
                let r = Math.random() * total;
                action = 0;
                for (let i = 0; i < weights.length; i++) {
                    r -= weights[i];
                    if (r <= 0) { action = i; break; }
                }
            } else {
                try {
                    const response = await this.pythonClient.predict(state, actionMask);
                    action = response.action;
                    qValues = response.qValues;
                    if (response.epsilon != null) this.epsilon = response.epsilon;
                } catch (err) {
                    wasRandom = true;
                    console.warn(`[Actor:${this.symbol}] Python error: ${err.message}, using random`);
                    const WEIGHTS = [2, 2, 94, 2];
                    const weights = actionMask.map((m, i) => m === 1 ? WEIGHTS[i] : 0);
                    const total = weights.reduce((a, b) => a + b, 0);
                    let r = Math.random() * total;
                    action = 0;
                    for (let i = 0; i < weights.length; i++) {
                        r -= weights[i];
                        if (r <= 0) { action = i; break; }
                    }
                }
            }

            const result = this.env.step(action);
            done = result.done;
            state = result.nextState;
            actionMask = result.actionMask;
            this.totalSteps++;

            episodeLog.push({
                price:        stepPrice,
                timestamp:    stepTimestamp,
                posBefore:    positionBefore,
                posAfter:     this.env.position ? this.env.position.side : null,
                posOpenPrice: this.env.position ? this.env.position.openPrice : null,
                action,
                wasRandom,
                qValues,
                reward:       result.reward,
                mask:         maskBeforeStep,
            });
        }

        // Pobierz doświadczenia zgodnie z return_mode
        const returnMode = (this.config.training.return_mode || 'mc').toLowerCase();
        const nStep      = this.config.training.n_step || 5;

        let experiences;
        let modeLabel;

        if (returnMode === 'nstep') {
            experiences = this.env.episode.getExperiencesNStep(nStep);
            modeLabel   = `${nStep}-step`;
        } else if (returnMode === 'td') {
            experiences = this.env.episode.getExperiencesTD();
            modeLabel   = 'TD';
        } else {
            // "mc" — zachowanie oryginalne
            experiences = this.env.getEpisodeExperiences();
            modeLabel   = 'MC';
        }

        this._printEpisodeLog(episodeLog, experiences);

        try {
            if (experiences.length > 0) {
                const batch = experiences.map(step => {
                    const entry = {
                        actorId:    this.symbol,
                        state:      step.state,
                        action:     step.action,
                        actionMask: step.actionMask,
                    };

                    if (returnMode === 'mc') {
                        // Oryginalny tryb: returnG jako reward, done=True wszędzie
                        entry.reward    = step.returnG;
                        entry.nextState = step.nextState;
                        entry.done      = true;
                    } else {
                        // nstep / td: prawdziwe done i nextState, gamma^n w polu gammaToN
                        entry.reward    = step.reward;
                        entry.nextState = step.nextState;
                        entry.done      = step.done;
                        entry.gammaToN  = step.gammaToN;
                        if (step.nextActionMask != null) {
                            entry.nextActionMask = step.nextActionMask;
                        }
                    }
                    return entry;
                });

                const batchResp = await this.pythonClient.sendBatch(batch);
                if (batchResp && batchResp.epsilon != null) this.epsilon = batchResp.epsilon;
                console.log(`[Actor:${this.symbol}] Episode ${this.totalEpisodes + 1}: sent ${batch.length} ${modeLabel} experiences, ε=${this.epsilon.toFixed(3)}`);
            }
        } catch (err) {
            console.warn(`[Actor:${this.symbol}] Batch send failed: ${err.message}`);
        }

        if (this.monitorClient) {
            try {
                const metrics = this.env.getMetrics();
                const lastTrade = this.env.getTrades().slice(-1)[0];
                let episodePnl = 0;
                if (lastTrade) {
                    const gross = lastTrade.side === 'SHORT'
                        ? (lastTrade.openPrice - lastTrade.closePrice) / lastTrade.openPrice
                        : (lastTrade.closePrice - lastTrade.openPrice) / lastTrade.openPrice;
                    const rc = this.config.reward;
                    episodePnl = gross - rc.commission_open - rc.commission_close - rc.trade_penalty;
                }
                
                // Track cumulative equity per episode
                this.cumulativePnl += episodePnl;
                this.episodeEquity = this.cumulativePnl;
                
                await this.monitorClient.sendMetrics(`actor:${this.symbol}`, {
                    epsilon: this.epsilon,
                    totalSteps: this.totalSteps,
                    totalEpisodes: this.totalEpisodes,
                    trades: this.env.getTrades().length,
                    win_rate: metrics.win_rate,
                    profit_factor: metrics.profit_factor,
                    episode_pnl: episodePnl,
                    cumulative_pnl: this.cumulativePnl
                });
            } catch (err) {
                // non-blocking
            }
        }
    }

    _printEpisodeLog(episodeLog, mcExperiences) {
        const A = ['LONG', 'SHORT', 'HOLD', 'CLOSE'];
        const metrics = this.env.getMetrics();
        const trades  = this.env.getTrades();

        // ANSI helpers
        const c = {
            green:  s => `\x1b[32m${s}\x1b[0m`,
            red:    s => `\x1b[31m${s}\x1b[0m`,
            yellow: s => `\x1b[33m${s}\x1b[0m`,
            cyan:   s => `\x1b[36m${s}\x1b[0m`,
            gray:   s => `\x1b[2m${s}\x1b[0m`,
            bold:   s => `\x1b[1m${s}\x1b[0m`,
        };
        // Kolor per akcja: LONG=zielony, SHORT=czerwony, HOLD=szary, CLOSE=żółty
        const ACTION_COLORS = [c.green, c.red, c.gray, c.yellow];

        const ep = this.totalEpisodes + 1;
        const rc = this.config.reward;
        let totalNetPnl = 0;
        for (const t of trades) {
            const gross = t.side === 'SHORT'
                ? (t.openPrice - t.closePrice) / t.openPrice
                : (t.closePrice - t.openPrice) / t.openPrice;
            totalNetPnl += gross - rc.commission_open - rc.commission_close - rc.trade_penalty;
        }

        const pnlStr = totalNetPnl >= 0
            ? c.green(`+${(totalNetPnl * 100).toFixed(3)}%`)
            : c.red(`${(totalNetPnl * 100).toFixed(3)}%`);
        console.log(`\n[${this.symbol}] Episode ${ep} | Steps: ${episodeLog.length} | Trades: ${trades.length} | W:${c.green(metrics.wins)} L:${c.red(metrics.losses)} | PnL: ${pnlStr} | ε=${this.epsilon.toFixed(3)}`);

        const table = new Table({
            head: ['#', 'Timestamp', 'Price', 'uPNL%', 'Pos', 'Src', 'Action', 'LONG', 'SHORT', 'HOLD', 'CLOSE', 'returnG'],
            colWidths: [5, 17, 11, 8, 5, 6, 21, 10, 10, 10, 10, 11],
            colAligns: ['right', 'left', 'right', 'right', 'middle', 'middle', 'left', 'right', 'right', 'right', 'right', 'right'],
            style: { head: [], border: [] },
        });

        const fmtQ = (q, blocked, chosen, actionIdx) => {
            if (blocked) return '';
            const marker = chosen ? '►' : ' ';
            const val = !isFinite(q) ? '-Inf' : q.toFixed(4);
            const str = `${marker}${val}`;
            return chosen
                ? c.bold(ACTION_COLORS[actionIdx](str))
                : ACTION_COLORS[actionIdx](str);
        };

        for (let i = 0; i < episodeLog.length; i++) {
            const row = episodeLog[i];
            const mc  = mcExperiences[i];

            const ts = row.timestamp
                ? new Date(row.timestamp).toISOString().slice(2, 16).replace('T', ' ')
                : '';

            const price = row.price != null ? row.price.toFixed(2) : '';

            // uPNL — niezrealizowany zysk otwartej pozycji
            let uPnl = '';
            if (row.posAfter && row.posOpenPrice && row.price != null) {
                const raw = row.posAfter === 'SHORT'
                    ? (row.posOpenPrice - row.price) / row.posOpenPrice
                    : (row.price - row.posOpenPrice) / row.posOpenPrice;
                const rawStr = `${raw >= 0 ? '+' : ''}${(raw * 100).toFixed(3)}`;
                uPnl = raw >= 0 ? c.green(rawStr) : c.red(rawStr);
            }

            const posBefore = row.posBefore ? row.posBefore[0] : '─';
            const posAfter  = row.posAfter  ? row.posAfter[0]  : '─';

            const src = row.wasRandom ? c.gray('rnd') : c.cyan('model');

            const rewardStr = row.reward !== 0
                ? ` r=${row.reward >= 0 ? c.green(`+${row.reward.toFixed(4)}`) : c.red(row.reward.toFixed(4))}`
                : '';
            const actionStr = `${ACTION_COLORS[row.action](A[row.action])}${rewardStr}`;

            const qCells = row.qValues
                ? row.qValues.map((q, ai) => fmtQ(q, row.mask[ai] === 0, ai === row.action, ai))
                : A.map((_, ai) => (row.mask[ai] === 0 ? '' : (ai === row.action ? c.bold('►') : '')));

            const returnGVal = mc ? mc.returnG : null;
            const returnG = returnGVal != null
                ? (returnGVal >= 0 ? c.green(`+${returnGVal.toFixed(5)}`) : c.red(returnGVal.toFixed(5)))
                : '';

            table.push([i + 1, ts, price, uPnl, `${posBefore}→${posAfter}`, src, actionStr, ...qCells, returnG]);
        }

        console.log(table.toString());
        console.log('');
    }

    stop() {
        this.running = false;
        this.binanceClient.destroy();
        console.log(`[Actor:${this.symbol}] STOPPED - Total: ${this.totalEpisodes} episodes, ${this.totalSteps} steps, ${this.env.getTrades().length} trades`);
    }
}

module.exports = { Actor };
