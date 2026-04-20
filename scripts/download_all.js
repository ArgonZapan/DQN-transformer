#!/usr/bin/env node
/**
 * Pobiera dane historyczne dla wszystkich symboli z config.toml × wszystkich aktywnych timeframe'ów.
 * Użycie: node scripts/download_all.js [--days 1500]
 */
const fs = require('fs');
const path = require('path');
const toml = require('toml');
const { downloadData } = require('./download_data');

const DEFAULT_DAYS = 1500; // ~4 lata + bufor

function parseArgs(argv) {
    let days = DEFAULT_DAYS;
    for (let i = 2; i < argv.length; i++) {
        if (argv[i] === '--days' && argv[i + 1]) {
            days = parseInt(argv[++i], 10);
        }
    }
    return { days };
}

async function main() {
    const { days } = parseArgs(process.argv);

    const configPath = path.join(__dirname, '..', 'config.toml');
    const config = toml.parse(fs.readFileSync(configPath, 'utf-8'));

    const symbols = config.actors.map(a => a.symbol);
    const tfCfg = config.timeframes;
    const intervals = Object.entries(tfCfg)
        .filter(([, v]) => v > 0)
        .map(([k]) => k.replace('candles_', ''));  // 'candles_1m' → '1m'

    const outputDir = config.data.path;

    console.log(`Symbole:    ${symbols.join(', ')}`);
    console.log(`Interwały:  ${intervals.join(', ')}`);
    console.log(`Dni:        ${days}`);
    console.log(`Katalog:    ${outputDir}`);
    console.log('');

    const tasks = [];
    for (const symbol of symbols) {
        for (const interval of intervals) {
            tasks.push({ symbol, interval });
        }
    }

    const CONCURRENCY = 50; // równoległe pobierania
    let done = 0;

    async function runTask({ symbol, interval }) {
        const n = ++done;
        console.log(`[${n}/${tasks.length}] START  ${symbol} ${interval}`);
        try {
            await downloadData(symbol, interval, days, outputDir);
            console.log(`[${n}/${tasks.length}] DONE   ${symbol} ${interval}`);
        } catch (err) {
            console.error(`[${n}/${tasks.length}] BŁĄD   ${symbol} ${interval}: ${err.message}`);
        }
    }

    // Pula workerów: zawsze CONCURRENCY zadań aktywnych jednocześnie
    const queue = [...tasks];
    const workers = Array.from({ length: CONCURRENCY }, async () => {
        while (queue.length > 0) {
            const task = queue.shift();
            if (task) await runTask(task);
        }
    });
    await Promise.all(workers);

    console.log('Pobieranie zakończone.');
}

main().catch(err => {
    console.error(err.message);
    process.exit(1);
});
