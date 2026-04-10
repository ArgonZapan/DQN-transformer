const { loadConfig } = require('./config');
const { ActorManager } = require('./actors/actorManager');

async function main() {
    const config = loadConfig();
    
    // Get actor symbol from command line argument (e.g., --actor=BTCUSDT)
    const actorArg = process.argv.find(arg => arg.startsWith('--actor='));
    const filterSymbol = actorArg ? actorArg.split('=')[1] : null;
    
    if (filterSymbol) {
        // Filter config to only run one actor
        config.actors = config.actors.filter(a => a.symbol === filterSymbol);
        if (config.actors.length === 0) {
            console.error(`[Node.js] Actor ${filterSymbol} not found in config`);
            process.exit(1);
        }
        console.log(`[Node.js] Running single actor: ${filterSymbol}`);
    }
    
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
