const fs   = require('fs');
const path = require('path');
const toml = require('toml');

const CONFIG_PATH = path.join(__dirname, '..', 'config.toml');

function loadConfig() {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8');
    const cfg = toml.parse(raw);

    if (!cfg.actors || cfg.actors.length === 0)
        throw new Error('config.toml: at least one [[actors]] entry required');
    if (!cfg.zmq)
        throw new Error('config.toml: [zmq] section required');
    if (!cfg.timeframes)
        throw new Error('config.toml: [timeframes] section required');
    if (!cfg.prediction)
        throw new Error('config.toml: [prediction] section required');
    if (!cfg.data)
        throw new Error('config.toml: [data] section required');

    return cfg;
}

module.exports = { loadConfig };
