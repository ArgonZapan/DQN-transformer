const https = require('https');
const fs = require('fs');
const path = require('path');
const toml = require('toml');

const BINANCE_BASE_URL = 'api.binance.com';
const KLINES_PATH = '/api/v3/klines';
const MAX_LIMIT = 1000;

const INTERVAL_MS = {
    '1m': 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000
};

function parseArgs(argv) {
    const args = {
        symbol: 'BTCUSDT',
        interval: '1h',
        days: 30,
        output: null
    };

    for (let i = 2; i < argv.length; i++) {
        // Check bounds before accessing next element
        if (i + 1 >= argv.length) break;
        
        switch (argv[i]) {
            case '--symbol':
                args.symbol = argv[++i]?.trim();
                break;
            case '--interval':
                args.interval = argv[++i]?.trim();
                break;
            case '--days':
                args.days = parseInt(argv[++i], 10) || 30;
                break;
            case '--output':
                args.output = argv[++i]?.trim();
                break;
        }
    }

    return args;
}

function loadOutputPath() {
    try {
        const configPath = path.join(__dirname, '..', 'config.toml');
        const content = fs.readFileSync(configPath, 'utf-8');
        const config = toml.parse(content);
        return config.data.path;
    } catch {
        return 'node/data/historical/';
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function fetchKlines(symbol, interval, startTime, endTime, limit) {
    return new Promise((resolve, reject) => {
        const params = new URLSearchParams({
            symbol,
            interval,
            startTime: startTime.toString(),
            endTime: endTime.toString(),
            limit: limit.toString()
        });

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
                    reject(new Error(`Failed to parse response: ${e.message}`));
                }
            });
        });

        req.on('error', reject);
        req.end();
    });
}

async function fetchWithRetry(symbol, interval, startTime, endTime, limit, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await fetchKlines(symbol, interval, startTime, endTime, limit);
        } catch (err) {
            if (attempt === maxRetries) {
                console.error(`Fetch failed after ${maxRetries} retries: ${err.message}`);
                throw err;
            }
            const delay = Math.pow(2, attempt - 1) * 1000;
            console.warn(`Binance API error (attempt ${attempt}/${maxRetries}), retry in ${delay}ms`);
            await sleep(delay);
        }
    }
}

function klineToCsvRow(kline) {
    return [
        kline[0],   // timestamp (open time)
        kline[1],   // open
        kline[2],   // high
        kline[3],   // low
        kline[4],   // close
        kline[5],   // volume
        kline[6],   // close_time
        kline[7],   // quote_volume
        kline[8],   // trades
        kline[9],   // taker_buy_base
        kline[10],  // taker_buy_quote
        kline[11]   // ignore
    ].join(',');
}

async function downloadData(symbol, interval, days, outputDir, concurrency = 50) {
    const intervalMs = INTERVAL_MS[interval];
    if (!intervalMs) {
        throw new Error(`Unsupported interval: ${interval}. Supported: ${Object.keys(INTERVAL_MS).join(', ')}`);
    }

    const resolvedOutput = path.resolve(path.join(__dirname, '..'), outputDir);
    const filepath = path.join(resolvedOutput, `${symbol}_${interval}.csv`);
    if (fs.existsSync(filepath)) {
        console.log(`  SKIP: ${symbol} ${interval} — plik już istnieje (${filepath})`);
        return filepath;
    }

    const endTime = Date.now();
    const startTime = endTime - (days * 24 * 60 * 60 * 1000);
    const chunkMs = MAX_LIMIT * intervalMs;

    // Wygeneruj wszystkie chunki z góry
    const chunks = [];
    for (let t = startTime; t < endTime; t += chunkMs) {
        chunks.push({ start: t, end: Math.min(t + chunkMs, endTime) });
    }

    const totalCandles = Math.ceil((endTime - startTime) / intervalMs);
    console.log(`Downloading ${symbol} ${interval} data: ~${totalCandles} candles (${days} days), ${chunks.length} requests, concurrency=${concurrency}`);

    if (!fs.existsSync(resolvedOutput)) {
        fs.mkdirSync(resolvedOutput, { recursive: true });
    }

    // Pula: zawsze `concurrency` żądań aktywnych jednocześnie
    const results = new Array(chunks.length);
    let dispatched = 0;
    let completed = 0;

    async function worker() {
        while (dispatched < chunks.length) {
            const idx = dispatched++;
            const { start, end } = chunks[idx];
            results[idx] = await fetchWithRetry(symbol, interval, start, end, MAX_LIMIT);
            completed++;
            if (completed % concurrency === 0 || completed === chunks.length) {
                const pct = (completed / chunks.length * 100).toFixed(1);
                console.log(`  Progress: ${pct}% (${completed}/${chunks.length} requests done)`);
            }
        }
    }

    await Promise.all(Array.from({ length: Math.min(concurrency, chunks.length) }, worker));

    const allKlines = results.flat();

    const deduplicated = [];
    const seen = new Set();
    for (const kline of allKlines) {
        const ts = kline[0];
        if (!seen.has(ts)) {
            seen.add(ts);
            deduplicated.push(kline);
        }
    }
    deduplicated.sort((a, b) => a[0] - b[0]);

    const header = 'timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore';
    const csvRows = [header, ...deduplicated.map(klineToCsvRow)];
    const csvContent = csvRows.join('\n') + '\n';

    fs.writeFileSync(filepath, csvContent, 'utf-8');

    console.log(`\nDone! Saved ${deduplicated.length} candles to ${filepath}`);
    console.log(`  Requests: ${chunks.length}`);
    console.log(`  Period: ${new Date(deduplicated[0][0]).toISOString()} — ${new Date(deduplicated[deduplicated.length - 1][0]).toISOString()}`);

    return filepath;
}

async function main() {
    const args = parseArgs(process.argv);

    if (!args.output) {
        args.output = loadOutputPath();
    }

    console.log(`Configuration:`);
    console.log(`  Symbol:   ${args.symbol}`);
    console.log(`  Interval: ${args.interval}`);
    console.log(`  Days:     ${args.days}`);
    console.log(`  Output:   ${args.output}`);
    console.log('');

    try {
        await downloadData(args.symbol, args.interval, args.days, args.output);
    } catch (err) {
        console.error(`Error: ${err.message}`);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { downloadData, fetchKlines, parseArgs, klineToCsvRow };
