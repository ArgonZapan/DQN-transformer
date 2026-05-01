const https = require('https');
const fs    = require('fs');
const path  = require('path');

const BINANCE_BASE_URL = 'api.binance.com';
const KLINES_PATH      = '/api/v3/klines';

const TF_TO_INTERVAL = {
    '1m': '1m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w',
};

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

class BinanceRateLimiter {
    constructor(config) {
        this.max = config.data.binance_rate_limit;
        this.count = 0;
        this._reset = setInterval(() => { this.count = 0; }, 60000);
    }
    async waitForSlot() {
        while (this.count >= this.max) await sleep(1000);
        this.count++;
    }
    destroy() { if (this._reset) { clearInterval(this._reset); this._reset = null; } }
}

function fetchKlinesRaw(symbol, interval, limit, startTime, endTime) {
    return new Promise((resolve, reject) => {
        const params = new URLSearchParams({ symbol, interval, limit: String(limit) });
        if (startTime) params.append('startTime', String(startTime));
        if (endTime)   params.append('endTime',   String(endTime));

        const req = https.request({
            hostname: BINANCE_BASE_URL,
            path: `${KLINES_PATH}?${params}`,
            method: 'GET',
            headers: { 'User-Agent': 'QuantileNet/1.0' },
        }, res => {
            let data = '';
            res.on('data', chunk => { data += chunk; });
            res.on('end', () => {
                if (res.statusCode !== 200) return reject(new Error(`Binance ${res.statusCode}: ${data}`));
                try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
            });
        });
        req.setTimeout(10000, () => req.destroy(new Error('timeout')));
        req.on('error', reject);
        req.end();
    });
}

function parseKline(raw) {
    return {
        timestamp:      parseInt(raw[0], 10),
        open:           parseFloat(raw[1]),
        high:           parseFloat(raw[2]),
        low:            parseFloat(raw[3]),
        close:          parseFloat(raw[4]),
        volume:         parseFloat(raw[5]),
        close_time:     parseInt(raw[6], 10),
        quote_volume:   parseFloat(raw[7]),
        trades:         parseInt(raw[8], 10),
        taker_buy_base: parseFloat(raw[9]),
        taker_buy_quote: parseFloat(raw[10]),
    };
}

function loadFromFile(symbol, tf, config) {
    const interval = TF_TO_INTERVAL[tf] || tf;
    const filename = `${symbol}_${interval}.csv`;
    const filepath = path.resolve(__dirname, '..', config.data.path, filename);

    if (!fs.existsSync(filepath))
        throw new Error(`Historical data file not found: ${filepath}`);

    const lines   = fs.readFileSync(filepath, 'utf-8').trim().split('\n');
    const candles = [];

    for (let i = 1; i < lines.length; i++) {
        const p = lines[i].split(',');
        if (p.length < 6) continue;
        candles.push({
            timestamp:       parseInt(p[0], 10),
            open:            parseFloat(p[1]),
            high:            parseFloat(p[2]),
            low:             parseFloat(p[3]),
            close:           parseFloat(p[4]),
            volume:          parseFloat(p[5]),
            close_time:      parseInt(p[6], 10) || 0,
            quote_volume:    parseFloat(p[7])   || 0,
            trades:          parseInt(p[8], 10) || 0,
            taker_buy_base:  parseFloat(p[9])   || 0,
            taker_buy_quote: parseFloat(p[10])  || 0,
        });
    }
    return candles;
}

async function loadCandles(symbol, tf, config) {
    if (config.data.source === 'file') {
        return loadFromFile(symbol, tf, config);
    }
    // API fallback (not the primary path)
    const interval = TF_TO_INTERVAL[tf] || tf;
    const raw = await fetchKlinesRaw(symbol, interval, 1000);
    return raw.map(parseKline);
}

module.exports = { loadCandles, BinanceRateLimiter, parseKline };
