const https = require('https');
const fs = require('fs');
const path = require('path');

const BINANCE_BASE_URL = 'api.binance.com';
const KLINES_PATH = '/api/v3/klines';

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

class BinanceRateLimiter {
    constructor(config) {
        this.maxRequestsPerMinute = config.data.binance_rate_limit;
        this.requestsThisMinute = 0;
        this._resetInterval = setInterval(() => { this.requestsThisMinute = 0; }, 60000);
    }

    async waitForSlot() {
        while (this.requestsThisMinute >= this.maxRequestsPerMinute) {
            await sleep(1000);
        }
        this.requestsThisMinute++;
    }

    destroy() {
        if (this._resetInterval) {
            clearInterval(this._resetInterval);
            this._resetInterval = null;
        }
    }
}

function fetchKlinesRaw(symbol, interval, limit, startTime, endTime) {
    return new Promise((resolve, reject) => {
        const params = new URLSearchParams({ symbol, interval, limit: limit.toString() });
        if (startTime) params.append('startTime', startTime.toString());
        if (endTime) params.append('endTime', endTime.toString());

        const options = {
            hostname: BINANCE_BASE_URL,
            path: `${KLINES_PATH}?${params.toString()}`,
            method: 'GET',
            headers: { 'User-Agent': 'TradingDQN/1.0' }
        };

        const req = https.request(options, (res) => {
            let data = '';
            res.on('data', chunk => { data += chunk; });
            res.on('end', () => {
                if (res.statusCode !== 200) {
                    reject(new Error(`Binance API error ${res.statusCode}: ${data}`));
                    return;
                }
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    reject(new Error(`Failed to parse Binance response: ${e.message}`));
                }
            });
        });

        req.setTimeout(10000, () => {
            req.destroy(new Error(`Binance request timeout for ${symbol} ${interval}`));
        });
        req.on('error', reject);
        req.end();
    });
}

function parseKline(raw) {
    return {
        timestamp: raw[0],
        open: parseFloat(raw[1]),
        high: parseFloat(raw[2]),
        low: parseFloat(raw[3]),
        close: parseFloat(raw[4]),
        volume: parseFloat(raw[5]),
        close_time: raw[6],
        quote_volume: parseFloat(raw[7]),
        trades: raw[8],
        taker_buy_base: parseFloat(raw[9]),
        taker_buy_quote: parseFloat(raw[10])
    };
}

class BinanceClient {
    constructor(config) {
        this.config = config;
        this.rateLimiter = new BinanceRateLimiter(config);
        this.dataSource = config.data.source;
        this.historicalPath = config.data.path;
        this.cachePath = config.data.cache_path;
        this.apiRetryInterval = config.data.api_retry_interval_sec * 1000;
    }

    async fetchKlines(symbol, interval, limit) {
        await this.rateLimiter.waitForSlot();
        const raw = await fetchKlinesRaw(symbol, interval, limit);
        return raw.map(parseKline);
    }

    async fetchWithRetry(symbol, interval, limit, maxRetries = 3) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await this.fetchKlines(symbol, interval, limit);
            } catch (err) {
                if (attempt === maxRetries) throw err;
                const delay = Math.pow(2, attempt - 1) * 1000;
                await sleep(delay);
            }
        }
    }

    loadFromFile(symbol, interval) {
        const filename = `${symbol}_${interval}.csv`;
        const filepath = path.resolve(this.historicalPath, filename);

        if (!fs.existsSync(filepath)) {
            throw new Error(`Historical data file not found: ${filepath}`);
        }

        const content = fs.readFileSync(filepath, 'utf-8');
        const lines = content.trim().split('\n');

        const header = lines[0];
        const candles = [];

        for (let i = 1; i < lines.length; i++) {
            const parts = lines[i].split(',');
            if (parts.length < 6) continue;

            candles.push({
                timestamp: parseInt(parts[0], 10),
                open: parseFloat(parts[1]),
                high: parseFloat(parts[2]),
                low: parseFloat(parts[3]),
                close: parseFloat(parts[4]),
                volume: parseFloat(parts[5]),
                close_time: parseInt(parts[6], 10) || 0,
                quote_volume: parseFloat(parts[7]) || 0,
                trades: parseInt(parts[8], 10) || 0,
                taker_buy_base: parseFloat(parts[9]) || 0,
                taker_buy_quote: parseFloat(parts[10]) || 0
            });
        }

        return candles;
    }

    async getData(symbol, interval, limit) {
        if (this.dataSource === 'file') {
            // Zwróć wszystkie dostępne świece — trening chce maksimum danych historycznych.
            // Limit jest ignorowany w trybie file (ma znaczenie tylko dla API real-time).
            return this.loadFromFile(symbol, interval);
        }
        return await this.fetchWithRetry(symbol, interval, limit);
    }

    destroy() {
        this.rateLimiter.destroy();
    }
}

module.exports = { BinanceClient, BinanceRateLimiter, parseKline };
