const fs = require('fs');
const path = require('path');
const toml = require('toml');

function loadConfig(configPath) {
    if (!configPath) {
        configPath = path.join(__dirname, '..', 'config.toml');
    }
    const content = fs.readFileSync(configPath, 'utf-8');
    return toml.parse(content);
}

module.exports = { loadConfig };
