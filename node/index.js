const { loadConfig } = require('./config');
const { ActorManager } = require('./actors/actorManager');

async function main() {
    const config = loadConfig();
    console.log(`[Node.js] Loaded config: ${config.actors.length} actors`);

    const manager = new ActorManager(config);

    async function gracefulShutdown() {
        console.log('[Node.js] Received shutdown signal — stopping actors...');
        await manager.stopAll();
        console.log('[Node.js] Actors stopped.');
        process.exit(0);
    }

    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);

    await manager.init();
    await manager.startAll();
}

main().catch(err => {
    console.error('[Node.js] Fatal error:', err);
    process.exit(1);
});
