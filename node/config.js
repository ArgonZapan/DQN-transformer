const fs = require('fs');
const path = require('path');
const toml = require('toml');

function loadConfig(configPath) {
    if (!configPath) {
        configPath = path.join(__dirname, '..', 'config.toml');
    }

    configPath = path.resolve(configPath);

    if (!fs.existsSync(configPath)) {
        throw new Error(`Config file not found: ${configPath}`);
    }

    const content = fs.readFileSync(configPath, 'utf-8');
    const config = toml.parse(content);

    validateConfig(config);

    return config;
}

function validateConfig(config) {
    const requiredSections = [
        'learner', 'training', 'model', 'features',
        'reward', 'data', 'logging', 'timeframes',
        'monitoring', 'per', 'backtesting', 'api'
    ];

    for (const section of requiredSections) {
        if (!(section in config)) {
            throw new Error(`Missing required config section: [${section}]`);
        }
    }

    if (!config.actors || config.actors.length === 0) {
        throw new Error('Config must have at least one [[actors]] entry');
    }

    const requiredLearner = ['host', 'port', 'metrics_port', 'device'];
    for (const key of requiredLearner) {
        if (!(key in config.learner)) {
            throw new Error(`Missing required learner config: ${key}`);
        }
    }

    const requiredTimeframes = ['candles_1m', 'candles_15m', 'candles_1h', 'candles_1d', 'candles_1w'];
    for (const key of requiredTimeframes) {
        if (!(key in config.timeframes)) {
            throw new Error(`Missing required timeframes config: ${key}`);
        }
    }
}

function getActorsConfig(config) {
    return config.actors;
}

function getTimeframesConfig(config) {
    return config.timeframes;
}

function getLearnerConfig(config) {
    return config.learner;
}

function getTrainingConfig(config) {
    return config.training;
}

function getMonitoringConfig(config) {
    return config.monitoring;
}

function getDataConfig(config) {
    return config.data;
}

function getRewardConfig(config) {
    return config.reward;
}

function getFeaturesConfig(config) {
    return config.features;
}

module.exports = {
    loadConfig,
    validateConfig,
    getActorsConfig,
    getTimeframesConfig,
    getLearnerConfig,
    getTrainingConfig,
    getMonitoringConfig,
    getDataConfig,
    getRewardConfig,
    getFeaturesConfig
};
