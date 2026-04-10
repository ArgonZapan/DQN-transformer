const zmq = require('zeromq');
const express = require('express');
const cors = require('cors');
const { loadConfig } = require('./config');

const config = loadConfig();
const monitoringConfig = config.monitoring;

const PORT = monitoringConfig.port;
const PULL_PORT = monitoringConfig.metrics_pull_port;

const metrics = {};

async function startZmqPull() {
    const sock = new zmq.Pull();
    await sock.bind(`tcp://*:${PULL_PORT}`);
    console.log(`[Monitoring] ZMQ PULL listening on port ${PULL_PORT}`);

    for await (const [msg] of sock) {
        try {
            const data = JSON.parse(msg.toString());
            const source = data.source || 'unknown';

            if (!metrics[source]) {
                metrics[source] = [];
                console.log(`[Monitoring] New source connected: ${source}`);
            }

            metrics[source].push({
                timestamp: data.timestamp || Date.now(),
                ...data.metrics
            });

            if (metrics[source].length > 1000) {
                metrics[source] = metrics[source].slice(-1000);
            }
            
            // Log every 100th message from each source
            if (metrics[source].length % 100 === 0) {
                console.log(`[Monitoring] ${source}: ${metrics[source].length} metrics received`);
            }
        } catch (err) {
            console.error(`[Monitoring] Parse error: ${err.message}`);
        }
    }
}

function startHttpServer() {
    const app = express();
    app.use(cors());

    app.get('/api/metrics', (req, res) => {
        res.json({
            timestamp: Date.now(),
            actors: Object.keys(metrics)
                .filter(k => k.startsWith('actor:'))
                .reduce((obj, k) => { obj[k] = metrics[k]; return obj; }, {}),
            learner: metrics['learner'] || []
        });
    });

    app.get('/api/status', (req, res) => {
        const lastLearner = metrics['learner']?.slice(-1)[0];
        const actorKeys = Object.keys(metrics).filter(k => k.startsWith('actor:'));
        const lastActorUpdate = actorKeys.length > 0
            ? Math.max(...actorKeys.map(k => metrics[k]?.slice(-1)?.[0]?.timestamp || 0))
            : 0;

        res.json({
            learnerConnected: !!lastLearner,
            actorsConnected: actorKeys.length,
            lastUpdate: Math.max(lastLearner?.timestamp || 0, lastActorUpdate)
        });
    });

    app.listen(PORT, () => {
        console.log(`[Monitoring] REST API on http://localhost:${PORT}`);
    });
}

async function gracefulShutdown() {
    console.log('[Monitoring] Shutting down...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    process.exit(0);
}

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

console.log('[Monitoring] 🚀 STARTING Monitoring Service...');
startZmqPull().catch(err => console.error('[Monitoring] ZMQ error:', err));
startHttpServer();

process.on('exit', () => {
    console.log('[Monitoring] 🛑 SHUTDOWN complete');
});
