const DataActor = require('./data_actor');

class ActorManager {
    constructor(config) {
        this.config = config;
        this.actors = [];
    }

    async init() {
        for (const actorCfg of this.config.actors) {
            const actor = new DataActor(actorCfg, this.config);
            this.actors.push(actor);
        }
        console.log(`[ActorManager] Initialized ${this.actors.length} data actor(s).`);
    }

    async runAll() {
        await Promise.all(this.actors.map(a => a.run()));
    }

    stop() {
        for (const actor of this.actors) actor.stop();
    }
}

module.exports = ActorManager;
