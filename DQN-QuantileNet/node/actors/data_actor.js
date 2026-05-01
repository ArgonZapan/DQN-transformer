const zmq      = require('zeromq');
const msgpack  = require('msgpack5')();
const { loadCandles }                    = require('../data/binance');
const { precomputeIndicators,
        assembleFeaturesFromPrecomputed,
        TIMEFRAME_KEYS,
        TIMEFRAME_CONFIG_KEYS }          = require('../env/state');

const BATCH_SIZE = 512;

// Ordered finest → coarsest; first active one becomes the stride base
const TF_ORDER = ['1m', '15m', '1h', '4h', '1d', '1w'];

// Minutes per timeframe — used to convert hour-horizons to base-TF steps
const TF_MINUTES = { '1m': 1, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080 };

class DataActor {
    constructor(actorCfg, config) {
        this.symbol  = actorCfg.symbol;
        this.config  = config;
        this.running = true;
        this.sock    = null;
    }

    stop() { this.running = false; }

    _activeTFs() {
        return TIMEFRAME_KEYS.filter(tf => (this.config.timeframes[TIMEFRAME_CONFIG_KEYS[tf]] || 0) > 0);
    }

    // Finest active TF — used as the iteration stride
    _baseTF() {
        const active = new Set(this._activeTFs());
        return TF_ORDER.find(tf => active.has(tf));
    }

    // Horizon hours → steps in base TF units
    _horizonSteps(baseTF) {
        const baseMin = TF_MINUTES[baseTF];
        return this.config.prediction.horizons_hours.map(h => Math.round((h * 60) / baseMin));
    }

    // For each base-TF candle index i, find the aligned end-index into each coarser TF
    _buildAlignments(allCandles, baseTF) {
        const baseCandles = allCandles[baseTF];
        const alignments  = {};

        for (const tf of this._activeTFs()) {
            if (tf === baseTF) continue;
            const tfCandles = allCandles[tf] || [];
            const align     = new Int32Array(baseCandles.length);
            let j = 0;
            for (let i = 0; i < baseCandles.length; i++) {
                const t = baseCandles[i].close_time;
                while (j < tfCandles.length && tfCandles[j].close_time <= t) j++;
                align[i] = j;
            }
            alignments[tf] = align;
        }
        return alignments;
    }

    // Build feature window for all active TFs at base-TF index i
    _buildWindow(precomputed, alignments, baseTF, i) {
        const state = {};
        for (const tf of this._activeTFs()) {
            const numCandles = this.config.timeframes[TIMEFRAME_CONFIG_KEYS[tf]];
            const endIdx     = tf === baseTF ? i + 1 : (alignments[tf] ? alignments[tf][i] : 0);
            const startIdx   = Math.max(0, endIdx - numCandles);
            state[tf]        = assembleFeaturesFromPrecomputed(precomputed[tf], startIdx, endIdx, numCandles);
        }
        return state;
    }

    async run() {
        const { host, port } = this.config.zmq;
        this.sock = new zmq.Push();
        await this.sock.connect(`${host}:${port}`);

        const baseTF    = this._baseTF();
        const activeTFs = this._activeTFs();

        if (!baseTF) {
            console.error(`[${this.symbol}] No active timeframes in config — skipping.`);
            return;
        }

        // Load candles for all active TFs
        const allCandles = {};
        for (const tf of activeTFs) {
            allCandles[tf] = await loadCandles(this.symbol, tf, this.config);
            console.log(`[${this.symbol}] Loaded ${allCandles[tf].length} candles @ ${tf}`);
        }

        const baseCandles = allCandles[baseTF];
        if (!baseCandles || baseCandles.length === 0) {
            console.error(`[${this.symbol}] No ${baseTF} candles — skipping.`);
            await this._sendDone(0);
            this.sock.close();
            return;
        }

        // Precompute indicators
        const normWindow  = this.config.features.normalization_window;
        const precomputed = {};
        for (const tf of activeTFs) {
            precomputed[tf] = precomputeIndicators(allCandles[tf], normWindow);
        }

        const alignments   = this._buildAlignments(allCandles, baseTF);
        const horizonSteps = this._horizonSteps(baseTF);
        const maxHorizon   = Math.max(...horizonSteps);

        // Warmup: enough candles for the largest input window in base-TF units
        const maxSeqInBase = Math.max(...activeTFs.map(tf => {
            const num  = this.config.timeframes[TIMEFRAME_CONFIG_KEYS[tf]];
            const ratio = TF_MINUTES[tf] / TF_MINUTES[baseTF];
            return Math.ceil(num * ratio);
        }));
        // Validation cutoff in base-TF steps
        const valDays   = this.config.training.validation_days || 30;
        const valSteps  = Math.round((valDays * 24 * 60) / TF_MINUTES[baseTF]);
        const endIdx    = baseCandles.length - maxHorizon - valSteps;

        // Training window: training_months = 0 means full history
        const trainMonths = this.config.training.training_months || 0;
        const trainSteps  = trainMonths > 0
            ? Math.round((trainMonths * 30 * 24 * 60) / TF_MINUTES[baseTF])
            : Infinity;
        const startIdx = trainMonths > 0
            ? Math.max(maxSeqInBase, endIdx - trainSteps)
            : maxSeqInBase;

        if (endIdx <= startIdx) {
            console.error(`[${this.symbol}] Not enough ${baseTF} candles for training window.`);
            await this._sendDone(0);
            this.sock.close();
            return;
        }

        const closes = baseCandles.map(c => c.close);
        let batch    = [];
        let totalSent = 0;

        for (let i = startIdx; i < endIdx && this.running; i++) {
            const futureCloses = horizonSteps.map(steps => {
                const fi = i + steps;
                return fi < closes.length ? closes[fi] : null;
            });
            if (futureCloses.some(v => v === null)) continue;

            batch.push({
                tf_data:       this._buildWindow(precomputed, alignments, baseTF, i),
                future_closes: futureCloses,
                current_close: closes[i],
                timestamp:     baseCandles[i].timestamp,
            });

            if (batch.length >= BATCH_SIZE) {
                await this._sendBatch(batch);
                totalSent += batch.length;
                batch = [];
                if (totalSent % 50000 === 0)
                    console.log(`[${this.symbol}] Sent ${totalSent} windows...`);
            }
        }

        if (batch.length > 0) {
            await this._sendBatch(batch);
            totalSent += batch.length;
        }

        await this._sendDone(totalSent);
        console.log(`[${this.symbol}] Done — sent ${totalSent} windows total.`);
        this.sock.close();
    }

    async _sendBatch(windows) {
        await this.sock.send(msgpack.encode({
            type:    'data_batch',
            symbol:  this.symbol,
            windows: windows,
        }));
    }

    async _sendDone(total) {
        await this.sock.send(msgpack.encode({
            type:   'done',
            symbol: this.symbol,
            total:  total,
        }));
    }
}

module.exports = DataActor;
