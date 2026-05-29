const { buildState, getActionMask, precomputeIndicators, buildStateFromPrecomputed, TIMEFRAME_KEYS, TIMEFRAME_CONFIG_KEYS } = require('./state');
const { calculateOpenPenalty, calculateReward, calculatePnLDecayed } = require('./reward');
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
        this.precomputed = null;
        this.tfAlignments = null;
        this.currentStepIndex = 0;
        this.trades = [];
        this.stepsSinceClose = Infinity; // kroki od ostatniego zamknięcia pozycji
        this.prevUnrealizedPnl = 0;     // poprzednie uPnL — do obliczania delty krok-po-kroku
        this._barsInTrade = 0;          // liczba kroków od otwarcia aktualnej pozycji
        this._actionCounts = [0, 0, 0, 0]; // [LONG, SHORT, HOLD, CLOSE] — zliczane co krok epizodu
        this._gapSteps = [];            // kroki od CLOSE do następnego OPEN (między transakcjami)
        this.minHoldSteps = config.training.min_hold_steps ?? 0;
    }

    setData(candlesPerTimeframe) {
        this.allCandles = candlesPerTimeframe;
        this._precompute();
    }

    _precompute() {
        const candles1m = this.allCandles['1m'];
        if (!candles1m || candles1m.length === 0) return;

        const t0 = Date.now();
        const normWindow = this.config.data.normalization_window;
        this.precomputed  = {};
        this.tfAlignments = {};

        for (const tf of TIMEFRAME_KEYS) {
            const configKey  = TIMEFRAME_CONFIG_KEYS[tf];
            const numCandles = this.config.timeframes[configKey];
            if (!numCandles || numCandles <= 0) continue;

            const candles = this.allCandles[tf] || [];
            if (candles.length === 0) continue;

            this.precomputed[tf] = precomputeIndicators(candles, normWindow);

            // Dla 1m endIdx = stepIndex+1, nie potrzeba mapy
            if (tf === '1m') continue;

            // Jednorazowy liniowy sweep O(N1m + NTF): dla każdej świecy 1m
            // znajdź ile świec TF jest <= close_time tej świecy
            const endIndices = new Int32Array(candles1m.length);
            let j = 0;
            for (let i = 0; i < candles1m.length; i++) {
                const t = candles1m[i].close_time;
                while (j < candles.length && candles[j].close_time <= t) j++;
                endIndices[i] = j;
            }
            this.tfAlignments[tf] = endIndices;
        }

        console.log(`[TradingEnv:${this.symbol}] Pre-computed indicators in ${Date.now() - t0}ms`);
    }

    getDataLength() {
        const tf1m = this.allCandles['1m'];
        return tf1m ? tf1m.length : 0;
    }

    reset() {
        this.position = null;
        this.trades = [];
        this.stepsSinceClose = Infinity;
        this.prevUnrealizedPnl = 0;
        this._barsInTrade = 0;
        this._actionCounts = [0, 0, 0, 0];
        this._gapSteps = [];
        const dataLen = this.getDataLength();
        this.currentStepIndex = this.episode.start(dataLen);
        return this._getState();
    }

    _getPositionFeatures(timestampMs) {
        // Cechy czasu: sin/cos godziny (okres 24h) + sin/cos dnia tygodnia (okres 7d) + is_weekend
        const t = new Date(timestampMs);
        const hourFrac   = (t.getUTCHours() + t.getUTCMinutes() / 60) / 24;
        const weekFrac   = (t.getUTCDay() * 24 + t.getUTCHours() + t.getUTCMinutes() / 60) / 168;
        const sinHour    = Math.sin(2 * Math.PI * hourFrac);
        const cosHour    = Math.cos(2 * Math.PI * hourFrac);
        const sinWeek    = Math.sin(2 * Math.PI * weekFrac);
        const cosWeek    = Math.cos(2 * Math.PI * weekFrac);
        const isWeekend  = (t.getUTCDay() === 0 || t.getUTCDay() === 6) ? 1.0 : 0.0;

        if (this.position === null) {
            return [0, 0, 0, 0, 0, sinHour, cosHour, sinWeek, cosWeek, isWeekend];
        }
        const currentPrice = this._getCurrentPrice();
        const isLong  = this.position.side === 'LONG'  ? 1 : 0;
        const isShort = this.position.side === 'SHORT' ? 1 : 0;
        const unrealizedPnl = isLong
            ? (currentPrice - this.position.openPrice) / this.position.openPrice
            : (this.position.openPrice - currentPrice) / this.position.openPrice;
        const stepInterval = this.config.training.step_interval || 15;
        const minutesHeld  = this._barsInTrade * stepInterval;
        const hold6h  = Math.min(minutesHeld / 360,  1.0);
        const hold48h = Math.min(minutesHeld / 2880, 1.0);
        return [isLong, isShort, unrealizedPnl, hold6h, hold48h, sinHour, cosHour, sinWeek, cosWeek, isWeekend];
    }

    _getState() {
        const candle1m = this.allCandles['1m'];
        if (!candle1m || this.currentStepIndex >= candle1m.length) return null;

        const currentTime = candle1m[this.currentStepIndex].close_time || Date.now();
        let state;
        if (this.precomputed) {
            state = buildStateFromPrecomputed(
                this.precomputed, this.tfAlignments,
                this.currentStepIndex, this.config
            );
        } else {
            state = buildState(this.allCandles, currentTime, this.config);
        }

        if (state) {
            state.position = this._getPositionFeatures(currentTime);
            // Required by Python QuantileFeatureLoader to look up prerolled
            // QuantileNet features for this (symbol, close_time_ms). Without
            // these, every lookup misses and emits warnings.
            state.timestamp = currentTime;
            state.symbol = this.symbol;
        }
        return state;
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

        // Capture the observation the agent actually acted on: the state BEFORE this
        // action mutates the position (and before the index advances). The position
        // features here reflect the pre-action position — exactly what was fed to the
        // model when it chose `action`. Capturing it after the mutation (as before)
        // desynchronized (state, action): an OPEN was stored with an already-in-position
        // state and a CLOSE with an already-flat one, so the network learned Q for the
        // wrong position context. With this in place, nextState = steps[t+n].state is
        // also pre-action, so the position-derived next-state mask in the learner is exact.
        const currentState = this._getState();

        // Rozbicie nagrody na składowe — do wyświetlania w tabeli aktora.
        // Wszystkie wartości są PRE-scale; reward_scale aplikowany na końcu do reward i rb łącznie.
        const rb = { open: 0, close: 0, close_pnl: 0, idle: 0, unreal: 0, delta: 0, hold: 0 };

        // Zlicz każdą akcję podjętą w epizodzie (używane przez getMetrics → statystyki)
        if (action >= 0 && action < 4) this._actionCounts[action]++;

        if (action === ACTIONS.LONG && this.position === null) {
            this.position = new Position('LONG', currentPrice, currentTime);
            this._barsInTrade = 0;
            if (this.stepsSinceClose !== Infinity) this._gapSteps.push(this.stepsSinceClose);
            rb.open = calculateOpenPenalty(this.rewardConfig, this.stepsSinceClose);
            reward += rb.open;
            this.stepsSinceClose = Infinity;
        } else if (action === ACTIONS.SHORT && this.position === null) {
            this.position = new Position('SHORT', currentPrice, currentTime);
            this._barsInTrade = 0;
            if (this.stepsSinceClose !== Infinity) this._gapSteps.push(this.stepsSinceClose);
            rb.open = calculateOpenPenalty(this.rewardConfig, this.stepsSinceClose);
            reward += rb.open;
            this.stepsSinceClose = Infinity;
        } else if (action === ACTIONS.CLOSE && this.position !== null) {
            this.position.holdSteps = this._barsInTrade;
            this.position.close(currentPrice, currentTime);
            rb.close = calculateReward(this.position, this.rewardConfig);
            rb.close_pnl = calculatePnLDecayed(this.position, this.rewardConfig);
            reward += rb.close;
            this.trades.push({ ...this.position });
            this.position = null;
            this.prevUnrealizedPnl = 0;
            this._barsInTrade = 0;
            this.stepsSinceClose = 0;
            if (this.maxTradesPerEpisode > 0 && this.trades.length >= this.maxTradesPerEpisode) {
                tradeClosed = true;
            }
        }

        // Licznik kroków od zamknięcia — inkrementuj przy każdym kroku gdy nie mamy pozycji
        if (this.position === null && this.stepsSinceClose < Infinity) {
            this.stepsSinceClose++;
        }

        // Kara za bezczynność (HOLD bez otwartej pozycji) — zwalcza action collapse.
        // Odejmowana przy każdym kroku bez pozycji, niezależnie od akcji.
        if (this.position === null) {
            const idlePenalty = this.rewardConfig.idle_penalty_per_bar ?? 0;
            if (idlePenalty > 0) {
                rb.idle = -idlePenalty;
                reward += rb.idle;
            }
        }

        // Intermediate reward: delta unrealized PnL w każdym kroku z otwartą pozycją.
        // Daje sygnał uczący przy HOLD zamiast reward=0 przez całe trzymanie pozycji.
        // Przy LONG/SHORT open: delta = 0 (openPrice = currentPrice → uPnl = 0 = prevUnrealizedPnl).
        // Przy CLOSE: reward już naliczony wyżej, pozycja null — tu nic nie robimy.
        if (this.position !== null) {
            const uPnl = this.position.side === 'LONG'
                ? (currentPrice - this.position.openPrice) / this.position.openPrice
                : (this.position.openPrice - currentPrice) / this.position.openPrice;
            const delta = uPnl - this.prevUnrealizedPnl;

            // Nagroda absolutna za unrealized PnL — daje kierunkowy sygnał per krok.
            const unrealizedScale = this.rewardConfig.unrealized_reward_scale ?? 0;
            if (unrealizedScale > 0) {
                rb.unreal = unrealizedScale * uPnl;
                reward += rb.unreal;
            }

            // Asymetryczne skalowanie: ujemne delty (spadki/boki) mnożone przez loss_scale.
            // Pozycja idąca bokiem kumuluje ujemne netto nawet jeśli PnL ≈ 0.
            const lossScale = this.rewardConfig.loss_scale ?? 1.0;
            const scaledDelta = delta < 0 ? delta * lossScale : delta;
            const cap = this.rewardConfig.intermediate_reward_max;
            rb.delta = Math.max(-cap, Math.min(cap, scaledDelta));
            reward += rb.delta;

            // Stały koszt trzymania pozycji — penalizuje długie trzymanie nawet przy flat PnL.
            const holdPenalty = this.rewardConfig.hold_penalty_per_bar ?? 0;
            if (holdPenalty > 0) {
                rb.hold = -holdPenalty;
                reward += rb.hold;
            }

            this.prevUnrealizedPnl = uPnl;
            this.position.updateDrawdown(currentPrice);
        }

        // Skalowanie całej nagrody — wzmacnia sygnał uczący (domyślnie 1.0 = brak zmiany).
        const rewardScale = this.rewardConfig.reward_scale ?? 1.0;
        reward *= rewardScale;
        for (const k of Object.keys(rb)) rb[k] *= rewardScale;

        this.currentStepIndex += this.stepInterval;

        // Increment barsInTrade after currentState is captured so nextState reflects N+1 steps held
        if (this.position !== null) {
            this._barsInTrade++;
        }

        const done = tradeClosed || this.episode.isAtTrainEnd(this.getDataLength()) || this._getState() === null;

        if (done && this.position !== null) {
            this.position.holdSteps = this._barsInTrade;
            this.position.close(currentPrice, currentTime);
            // Wymuś zamknięcie — nadpisz całą nagrodę i rb
            for (const k of Object.keys(rb)) rb[k] = 0;
            rb.close = calculateReward(this.position, this.rewardConfig) * rewardScale;
            rb.close_pnl = calculatePnLDecayed(this.position, this.rewardConfig) * rewardScale;
            reward = rb.close;
            this.trades.push({ ...this.position });
            this.position = null;
            this.prevUnrealizedPnl = 0;
            this._barsInTrade = 0;
        }

        const nextState = done ? null : this._getState();
        const actionMask = this.getActionMask();

        this.episode.addStep(
            currentState,
            action,
            reward,
            nextState,
            done,
            actionMask
        );

        return { nextState, reward, done, actionMask, rewardBreakdown: rb };
    }

    getActionMask() {
        const mask = getActionMask(this.position);
        // Blokuj CLOSE przez pierwsze minHoldSteps kroków od otwarcia pozycji.
        if (this.position !== null && this.minHoldSteps > 0 && this._barsInTrade < this.minHoldSteps) {
            mask[3] = 0;
        }
        return mask;
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
        const ac = this._actionCounts;

        if (trades.length === 0) {
            return {
                win_rate: 0, profit_factor: 0, total_trades: 0, net_pnl: 0,
                max_consecutive_losses: 0, wins: 0, losses: 0,
                profit_sum: 0, loss_sum: 0,
                win_hold_steps: 0, loss_hold_steps: 0,
                mfe_sum: 0,
                gap_steps: [],
                long_opens:    ac[0], short_opens: ac[1],
                action_counts: [ac[0], ac[1], ac[2], ac[3]],
            };
        }

        let wins = 0, losses = 0;
        let totalProfit = 0, totalLoss = 0;
        let currentConsec = 0, maxConsec = 0;
        let winHoldSteps = 0, lossHoldSteps = 0;
        let mfeSum = 0;

        const commission  = this.rewardConfig.commission_open + this.rewardConfig.commission_close;
        const tradePenalty = this.rewardConfig.trade_penalty;

        for (const trade of trades) {
            const gross = trade.side === 'SHORT'
                ? (trade.openPrice - trade.closePrice) / trade.openPrice
                : (trade.closePrice - trade.openPrice) / trade.openPrice;
            const pnl       = gross - commission - tradePenalty;
            const holdSteps = trade.holdSteps || 0;
            mfeSum += (trade.peakPnl || 0);

            if (pnl > 0) {
                wins++;
                totalProfit  += pnl;
                winHoldSteps += holdSteps;
                currentConsec = 0;
            } else {
                losses++;
                totalLoss    += Math.abs(pnl);
                lossHoldSteps += holdSteps;
                currentConsec++;
                if (currentConsec > maxConsec) maxConsec = currentConsec;
            }
        }

        const winRate      = wins / trades.length;
        const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : (totalProfit > 0 ? 999 : 0);

        return {
            win_rate:               winRate,
            profit_factor:          profitFactor,
            total_trades:           trades.length,
            wins,
            losses,
            net_pnl:                totalProfit - totalLoss,
            max_consecutive_losses: maxConsec,
            profit_sum:             totalProfit,
            loss_sum:               totalLoss,
            win_hold_steps:         winHoldSteps,
            loss_hold_steps:        lossHoldSteps,
            mfe_sum:                mfeSum,
            gap_steps:              this._gapSteps.slice(),
            long_opens:             ac[0],
            short_opens:            ac[1],
            action_counts:          [ac[0], ac[1], ac[2], ac[3]],
        };
    }
}

module.exports = { TradingEnv, Position, ACTIONS };
