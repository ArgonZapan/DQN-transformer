const { Actor } = require('./actor');
const { PythonClient } = require('../client/pythonClient');
const { MonitorClient } = require('../monitoring/monitor_client');

class ActorManager {
    constructor(config) {
        this.config = config;
        this.actors = [];
        this.pythonClient = new PythonClient(config);
        this.monitorClient = new MonitorClient(config);
    }

    async init() {
        await this.pythonClient.connect();

        try {
            await this.monitorClient.connect();
        } catch (err) {
            console.warn(`[ActorManager] Could not connect to monitoring: ${err.message}`);
        }

        for (const actorConfig of this.config.actors) {
            const actor = new Actor(
                actorConfig,
                this.config,
                this.pythonClient,
                this.monitorClient
            );
            await actor.loadData();
            this.actors.push(actor);
            console.log(`[ActorManager] Initialized actor for ${actorConfig.symbol}`);
        }
    }

    async startAll() {
        console.log(`[ActorManager] Starting ${this.actors.length} actors...`);
        const promises = this.actors.map(actor => actor.start());
        await Promise.all(promises);
    }

    async stopAll() {
        console.log('[ActorManager] Stopping all actors...');
        for (const actor of this.actors) {
            actor.stop();
        }
        this.pythonClient.close();
        this.monitorClient.close();
        console.log('[ActorManager] All actors stopped.');
    }
}

module.exports = { ActorManager };
