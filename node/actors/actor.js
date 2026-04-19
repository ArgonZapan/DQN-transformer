const fs    = require('fs');
const Table = require('cli-table3');
const { TradingEnv } = require('../env/tradingEnv');
const { BinanceClient } = require('../data/binance');
const { PerPairNormalizer } = require('../data/normalizer');

const TIMEFRAME_INTERVALS = ['1m', '15m', '1h', '1d', '1w'];

// Lazy-load onnxruntime-node — brak pakietu nie blokuje startu aktora
let ort = null;
try {
    ort = require('onnxruntime-node');
} catch (_) {
    // onnxruntime-node niedostępny — aktor użyje RPC
}

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
        this._lastEpisodeLogTime = 0;  // throttle dla linii informacyjnej w start()
        this._lastTableLogTime   = 0;  // throttle dla tabeli epizodu (niezależny)
        this.episodeMirror = globalConfig.training.episode_mirror ?? false;

        // ONNX local inference state
        this._onnxSession = null;
        this._onnxMtime   = 0;
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

        this._applyDataWindow(candlesPerTf);
        this.env.setData(candlesPerTf);
    }

    /**
     * Obcina dane do okna wyznaczonego przez training_months + validation_weeks/days.
     * Zostawia też bufor rozgrzewki (candles_1d dni) przed treningiem — sieć potrzebuje
     * pełnego okna 1d na początku danych treningowych.
     * Tylko tryb file; w trybie API dane już są ograniczone do N ostatnich świec.
     */
    _applyDataWindow(candlesPerTf) {
        const trainingMonths  = this.config.training.training_months  || 0;
        const validationWeeks = this.config.training.validation_weeks || 0;
        const validationDays  = validationWeeks
            ? validationWeeks * 7
            : this.config.training.validation_days;
        // Warmup = okno 1d-owe (największy TF) — tyle dni musi być przed pierwszym epizod
        const warmupDays = this.config.timeframes.candles_1d || 0;

        const candles1m = candlesPerTf['1m'];
        if (!candles1m || candles1m.length === 0) return;

        const msPerDay = 24 * 60 * 60 * 1000;
        const lastTs   = candles1m[candles1m.length - 1].timestamp;
        const firstTs  = candles1m[0].timestamp;
        const totalDaysRaw = Math.round((lastTs - firstTs) / msPerDay);

        if (trainingMonths > 0) {
            const valMs    = validationDays    * msPerDay;
            const trainMs  = trainingMonths * 30 * msPerDay;
            const warmupMs = warmupDays        * msPerDay;
            const cutStart = lastTs - valMs - trainMs - warmupMs;

            const countBefore = candles1m.length;
            for (const tf of Object.keys(candlesPerTf)) {
                candlesPerTf[tf] = candlesPerTf[tf].filter(c => c.timestamp >= cutStart);
            }
            const countAfter = (candlesPerTf['1m'] || []).length;

            console.log(`[Actor:${this.symbol}] ── Okno danych (training_months=${trainingMonths}) ──`);
            console.log(`  Rozgrzewka (buffor sieci):  ${warmupDays} dni`);
            console.log(`  Trening:                    ${trainingMonths * 30} dni (~${trainingMonths} mies.)`);
            console.log(`  Walidacja:                  ${validationDays} dni${validationWeeks ? ` (${validationWeeks} tyg.)` : ''}`);
            console.log(`  Łącznie w oknie:            ${warmupDays + trainingMonths * 30 + validationDays} dni`);
            console.log(`  Świece 1m: ${countBefore} → ${countAfter} (odcięto ${countBefore - countAfter})`);
        } else {
            // Brak obcinania treningu — pokaż tylko statystyki
            const trainDays = totalDaysRaw - validationDays;
            console.log(`[Actor:${this.symbol}] Dane: ${totalDaysRaw} dni łącznie`);
            console.log(`  Trening: ~${trainDays} dni | Walidacja: ${validationDays} dni${validationWeeks ? ` (${validationWeeks} tyg.)` : ''} | Rozgrzewka: ${warmupDays} dni (buffor sieci)`);
        }
    }

    async start() {
        this.running = true;
        this._startTime = Date.now();
        console.log(`[Actor:${this.symbol}] Started`);
        while (this.running) {
            try {
                const episodeStart = Date.now();
                const stepsBefore = this.totalSteps;
                await this.runEpisode();
                this.totalEpisodes++;
                const episodeTime = Date.now() - episodeStart;
                const episodeSteps = this.totalSteps - stepsBefore;
                const episodeSps = episodeTime > 0 ? (episodeSteps / (episodeTime / 1000)).toFixed(0) : '?';
                const totalSps = this._startTime ? (this.totalSteps / ((Date.now() - this._startTime) / 1000)).toFixed(1) : '?';
                const _now = Date.now();
                if (_now - this._lastEpisodeLogTime >= 5000) {
                    this._lastEpisodeLogTime = _now;
                    console.log(`[Actor:${this.symbol}] Episode ${this.totalEpisodes} | ${episodeTime}ms | steps=${this.totalSteps} | sps=${episodeSps} (avg ${totalSps}) | trades=${this.env.getTrades().length}`);
                }
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
                // 1. Spróbuj lokalnego wnioskowania ONNX (brak latencji TCP)
                const onnxResult = await this._inferOnnx(state, actionMask);
                if (onnxResult) {
                    action  = onnxResult.action;
                    qValues = onnxResult.qValues;
                } else {
                    // 2. Fallback: RPC do Python
                    try {
                        const response = await this.pythonClient.predict(state, actionMask);
                        action  = response.action;
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
            }

            const result = this.env.step(action);
            done = result.done;
            state = result.nextState;
            actionMask = result.actionMask;
            this.totalSteps++;

            episodeLog.push({
                price:          stepPrice,
                timestamp:      stepTimestamp,
                posBefore:      positionBefore,
                posAfter:       this.env.position ? this.env.position.side : null,
                posOpenPrice:   this.env.position ? this.env.position.openPrice : null,
                action,
                wasRandom,
                qValues,
                reward:         result.reward,
                rewardBreakdown: result.rewardBreakdown,
                mask:           maskBeforeStep,
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

        const mirrorEnabled = this.episodeMirror;
        let mirrorExperiences = null;
        let mirrorEpisodeLog = null;
        if (mirrorEnabled) {
            mirrorExperiences = this._buildMirrorExperiences(episodeLog, returnMode, nStep);
            mirrorEpisodeLog  = this._buildMirrorEpisodeLog(episodeLog, mirrorExperiences);
        }

        const _now = Date.now();
        if (_now - this._lastTableLogTime >= 5000) {
            this._lastTableLogTime = _now;
            this._printEpisodeLog(episodeLog, experiences, mirrorEnabled ? 'ORYGINAŁ' : '');
            if (mirrorEnabled && mirrorExperiences) {
                this._printEpisodeLog(mirrorEpisodeLog, mirrorExperiences, 'MIRROR');
            }
        }

        try {
            if (experiences.length > 0) {
                const _toEntry = (step) => {
                    const entry = {
                        actorId:    this.symbol,
                        state:      step.state,
                        action:     step.action,
                        actionMask: step.actionMask,
                    };
                    if (returnMode === 'mc') {
                        entry.reward    = step.returnG;
                        entry.nextState = step.nextState;
                        entry.done      = true;
                    } else {
                        entry.reward    = step.reward;
                        entry.nextState = step.nextState;
                        entry.done      = step.done;
                        entry.gammaToN  = step.gammaToN;
                        if (step.nextActionMask != null) {
                            entry.nextActionMask = step.nextActionMask;
                        }
                    }
                    return entry;
                };

                const batch = experiences.map(step => _toEntry(step));

                // Dodaj metryki epizodu do ostatniego wpisu — learner zbiera je osobno
                const epMetrics = this.env.getMetrics();
                const ac = epMetrics.action_counts || [0, 0, 0, 0];
                batch[batch.length - 1].metrics = {
                    episode_length:            experiences.length,
                    episode_trades:            epMetrics.total_trades,
                    episode_wins:              epMetrics.wins              || 0,
                    episode_losses:            epMetrics.losses            || 0,
                    episode_pnl:               epMetrics.net_pnl           || 0,
                    episode_max_consec_losses: epMetrics.max_consecutive_losses || 0,
                    // Statystyki handlowe
                    episode_profit_sum:        epMetrics.profit_sum        || 0,
                    episode_loss_sum:          epMetrics.loss_sum          || 0,
                    episode_win_hold_steps:    epMetrics.win_hold_steps    || 0,
                    episode_loss_hold_steps:   epMetrics.loss_hold_steps   || 0,
                    episode_mfe_sum:           epMetrics.mfe_sum           || 0,
                    episode_gap_steps:         epMetrics.gap_steps         || [],
                    episode_long_opens:        epMetrics.long_opens        || 0,
                    episode_short_opens:       epMetrics.short_opens       || 0,
                    // Rozkład akcji
                    episode_action_long:       ac[0] || 0,
                    episode_action_short:      ac[1] || 0,
                    episode_action_hold:       ac[2] || 0,
                    episode_action_close:      ac[3] || 0,
                };

                if (mirrorEnabled && mirrorExperiences) {
                    for (const step of mirrorExperiences) {
                        batch.push(_toEntry(step));
                    }
                }

                const batchResp = await this.pythonClient.sendBatch(batch);
                if (batchResp && batchResp.epsilon != null) this.epsilon = batchResp.epsilon;
                if (batchResp && batchResp.loss_scale != null) this.env.rewardConfig.loss_scale = batchResp.loss_scale;
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

    // ── Episode mirroring helpers ──────────────────────────────────────────

    _mirrorActionMask(mask) {
        // Swap LONG(0) ↔ SHORT(1), keep HOLD(2) and CLOSE(3)
        return [mask[1], mask[0], mask[2], mask[3]];
    }

    _mirrorStateFeatures(state) {
        if (!state) return null;
        const mirrored = {};
        for (const [tf, rows] of Object.entries(state)) {
            if (tf === 'position') continue;
            mirrored[tf] = rows.map(row => {
                const r = [...row];
                // [0]=normalizedClose negate, [1]=relativeRange unchanged,
                // [2]=candleDirection negate, [3]=normalizedVolume unchanged,
                // [4]=rsiNorm → 1-v, [5]=macdNorm negate,
                // [6]=pctChange negate, [7]=aboveSma → 1-v
                r[0] = -r[0];
                r[2] = -r[2];
                r[4] = 1 - r[4];
                r[5] = -r[5];
                r[6] = -r[6];
                r[7] = 1 - r[7];
                return r;
            });
        }
        if (state.position) {
            const [isLong, isShort, upnl, barsNorm] = state.position;
            mirrored.position = [isShort, isLong, -upnl, barsNorm];
        }
        return mirrored;
    }

    _buildMirrorExperiences(episodeLog, returnMode, nStep) {
        const steps = this.env.episode.steps;
        const T = steps.length;
        const gamma = this.config.training.gamma;
        const gammaN = Math.pow(gamma, nStep);

        // Mirrored immediate rewards: flip pnl direction via close_pnl, negate delta/unreal
        const mirRew = episodeLog.map(row => {
            const rb = row.rewardBreakdown || {};
            const closeMirror = (rb.close || 0) - 2 * (rb.close_pnl || 0);
            return (rb.open || 0) + closeMirror + (rb.idle || 0)
                 - (rb.delta || 0) - (rb.unreal || 0) + (rb.hold || 0);
        });

        // MC returns from mirrored rewards (for display)
        let G = 0;
        const mcReturns = new Array(T);
        for (let t = T - 1; t >= 0; t--) {
            G = mirRew[t] + gamma * G;
            mcReturns[t] = G;
        }

        const mirAction = a => (a === 0 ? 1 : (a === 1 ? 0 : a));

        if (returnMode === 'mc') {
            return steps.map((step, t) => ({
                state:      this._mirrorStateFeatures(step.state),
                action:     mirAction(step.action),
                reward:     mcReturns[t],
                nextState:  step.nextState ? this._mirrorStateFeatures(step.nextState) : null,
                done:       true,
                actionMask: this._mirrorActionMask(step.actionMask),
                returnG:    mcReturns[t],
                gammaToN:   gamma,
            }));
        } else if (returnMode === 'td') {
            return steps.map((step, t) => ({
                state:      this._mirrorStateFeatures(step.state),
                action:     mirAction(step.action),
                reward:     mirRew[t],
                nextState:  step.nextState ? this._mirrorStateFeatures(step.nextState) : null,
                done:       step.done,
                actionMask: this._mirrorActionMask(step.actionMask),
                returnG:    mcReturns[t],
                gammaToN:   gamma,
            }));
        } else {
            // nstep
            const result = [];
            for (let t = 0; t < T; t++) {
                let reward = 0;
                for (let k = 0; k < nStep && (t + k) < T; k++) {
                    reward += Math.pow(gamma, k) * mirRew[t + k];
                }
                const endIdx = t + nStep;
                const done = endIdx >= T;
                const nextStep = done ? null : steps[endIdx];
                result.push({
                    state:          this._mirrorStateFeatures(steps[t].state),
                    action:         mirAction(steps[t].action),
                    reward,
                    nextState:      nextStep ? this._mirrorStateFeatures(nextStep.state) : null,
                    done,
                    actionMask:     this._mirrorActionMask(steps[t].actionMask),
                    nextActionMask: nextStep ? this._mirrorActionMask(nextStep.actionMask) : null,
                    gammaToN:       gammaN,
                    returnG:        mcReturns[t],
                });
            }
            return result;
        }
    }

    _buildMirrorEpisodeLog(episodeLog, mirrorExperiences) {
        const mirPos = p => (p === 'LONG' ? 'SHORT' : (p === 'SHORT' ? 'LONG' : p));
        const mirAction = a => (a === 0 ? 1 : (a === 1 ? 0 : a));
        return episodeLog.map((row, i) => {
            const rb = row.rewardBreakdown || {};
            const closePnl = rb.close_pnl || 0;
            const mirRb = {
                open:      rb.open   || 0,
                close:     (rb.close || 0) - 2 * closePnl,
                close_pnl: -closePnl,
                idle:      rb.idle   || 0,
                delta:     -(rb.delta  || 0),
                unreal:    -(rb.unreal || 0),
                hold:      rb.hold   || 0,
            };
            const mirReward = mirRb.open + mirRb.close + mirRb.idle + mirRb.delta + mirRb.unreal + mirRb.hold;
            return {
                price:           row.price,
                timestamp:       row.timestamp,
                posBefore:       mirPos(row.posBefore),
                posAfter:        mirPos(row.posAfter),
                posOpenPrice:    row.posOpenPrice,
                action:          mirAction(row.action),
                wasRandom:       row.wasRandom,
                qValues:         row.qValues ? [row.qValues[1], row.qValues[0], row.qValues[2], row.qValues[3]] : null,
                reward:          mirReward,
                rewardBreakdown: mirRb,
                mask:            this._mirrorActionMask(row.mask),
            };
        });
    }

    _printEpisodeLog(episodeLog, mcExperiences, label = '') {

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
        const labelStr = label ? ` ── ${label}` : '';
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
        const avgSps = this._startTime && this.totalSteps > 0
            ? (this.totalSteps / ((Date.now() - this._startTime) / 1000)).toFixed(1)
            : '?';
        console.log(`\n[${this.symbol}] Episode ${ep}${labelStr} | Steps: ${episodeLog.length} | Trades: ${trades.length} | W:${c.green(metrics.wins)} L:${c.red(metrics.losses)} | PnL: ${pnlStr} | ε=${this.epsilon.toFixed(3)} | sps=${avgSps}`);

        const table = new Table({
            head: ['#', 'Timestamp', 'Price', 'uPNL%', 'Pos', 'Src', 'Action',
                   'LONG', 'SHORT', 'HOLD', 'CLOSE',
                   'rOpen', 'rClose', 'rIdle', 'rΔuPnL', 'ruPnL', 'rHold', 'rΣ',
                   'r5step'],
            colWidths: [5, 17, 11, 8, 5, 6, 12, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 10, 11],
            colAligns: ['right', 'left', 'right', 'right', 'middle', 'middle', 'left',
                        'right', 'right', 'right', 'right',
                        'right', 'right', 'right', 'right', 'right', 'right', 'right',
                        'right'],
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

        // Formatuj składową nagrody: puste gdy zero, kolorowe gdy niezerowe
        const fmtR = v => {
            if (v === 0 || v == null) return '';
            const s = v >= 0 ? `+${v.toFixed(4)}` : v.toFixed(4);
            return v >= 0 ? c.green(s) : c.red(s);
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
            const actionStr = ACTION_COLORS[row.action](A[row.action]);

            const qCells = row.qValues
                ? row.qValues.map((q, ai) => fmtQ(q, row.mask[ai] === 0, ai === row.action, ai))
                : A.map((_, ai) => (row.mask[ai] === 0 ? '' : (ai === row.action ? c.bold('►') : '')));

            const rb = row.rewardBreakdown || {};
            const rTotal = fmtR(row.reward);

            const nstepVal = mc ? mc.reward : null;
            const returnG = nstepVal != null
                ? (nstepVal >= 0 ? c.green(`+${nstepVal.toFixed(5)}`) : c.red(nstepVal.toFixed(5)))
                : '';

            table.push([
                i + 1, ts, price, uPnl, `${posBefore}→${posAfter}`, src, actionStr,
                ...qCells,
                fmtR(rb.open), fmtR(rb.close), fmtR(rb.idle),
                fmtR(rb.delta), fmtR(rb.unreal), fmtR(rb.hold), rTotal,
                returnG,
            ]);
        }

        console.log(table.toString());
        console.log('');
    }

    // ── ONNX local inference ────────────────────────────────────────────────

    /**
     * Zwraca aktywną sesję ONNX (wczytuje lub przeładowuje gdy plik się zmienił).
     * Zwraca null gdy ONNX wyłączony, pakiet niedostępny lub plik nie istnieje jeszcze.
     */
    async _getOnnxSession() {
        if (!ort) return null;
        const onnxCfg = this.config.onnx || {};
        if (!onnxCfg.enabled) return null;

        const modelPath = onnxCfg.export_path || 'python/checkpoints/model.onnx';
        try {
            const stat  = fs.statSync(modelPath);
            const mtime = stat.mtimeMs;
            if (!this._onnxSession || mtime > this._onnxMtime) {
                const providers = (onnxCfg.device === 'cuda')
                    ? ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    : ['CPUExecutionProvider'];
                if (this._onnxSession) {
                    await this._onnxSession.release?.();
                    this._onnxSession = null;
                }
                this._onnxSession = await ort.InferenceSession.create(modelPath, { executionProviders: providers });
                this._onnxMtime   = mtime;
                console.log(`[Actor:${this.symbol}] ONNX model loaded (step ~${this.totalSteps}, mtime: ${new Date(mtime).toISOString()})`);
            }
            return this._onnxSession;
        } catch (_) {
            return null;  // model jeszcze nie wyeksportowany
        }
    }

    /**
     * Lokalne wnioskowanie przez ONNX — O(0.1–0.5 ms) zamiast 1–5 ms TCP.
     * Zwraca { action, qValues } lub null gdy ONNX niedostępny.
     */
    async _inferOnnx(state, actionMask) {
        const session = await this._getOnnxSession();
        if (!session) return null;

        try {
            const tfKeys      = Object.keys(this.config.timeframes)
                .filter(k => this.config.timeframes[k] > 0)
                .sort();
            const numFeatures = this.config.features.num_features;

            const feeds = {};

            for (let i = 0; i < tfKeys.length; i++) {
                const key    = tfKeys[i];
                const tfName = key.replace('candles_', '');
                const seqLen = this.config.timeframes[key];
                const rows   = (state && state[tfName]) || [];

                const arr = new Float32Array(seqLen * numFeatures);
                const len = Math.min(rows.length, seqLen);
                for (let s = 0; s < len; s++) {
                    const row = rows[s] || [];
                    for (let f = 0; f < numFeatures; f++) {
                        arr[s * numFeatures + f] = row[f] || 0;
                    }
                }
                feeds[`tf_${i}`] = new ort.Tensor('float32', arr, [1, seqLen, numFeatures]);
            }

            const posData = (state && state.position) || [0, 0, 0, 0];
            feeds['pos_features'] = new ort.Tensor('float32', new Float32Array(posData.slice(0, 4)), [1, 4]);
            feeds['action_mask']  = new ort.Tensor('float32', new Float32Array(actionMask), [1, 4]);

            const results = await session.run(feeds);
            const qRaw    = Array.from(results['q_values'].data);

            // Wybierz najlepszą dozwoloną akcję
            let bestAction = -1, bestQ = -Infinity;
            for (let a = 0; a < qRaw.length; a++) {
                if (actionMask[a] === 1 && qRaw[a] > bestQ) {
                    bestQ = qRaw[a];
                    bestAction = a;
                }
            }
            if (bestAction === -1) bestAction = actionMask.indexOf(1);  // fallback

            return { action: bestAction, qValues: qRaw };

        } catch (err) {
            console.warn(`[Actor:${this.symbol}] ONNX inference error: ${err.message}`);
            this._onnxSession = null;  // wymuś przeładowanie przy następnej próbie
            return null;
        }
    }

    stop() {
        this.running = false;
        this.binanceClient.destroy();
        console.log(`[Actor:${this.symbol}] STOPPED - Total: ${this.totalEpisodes} episodes, ${this.totalSteps} steps, ${this.env.getTrades().length} trades`);
    }
}

module.exports = { Actor };
