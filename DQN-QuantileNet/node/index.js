const { loadConfig }   = require('./config');
const ActorManager     = require('./actors/actor_manager');

async function main() {
    const config = loadConfig();
    const manager = new ActorManager(config);

    process.on('SIGTERM', () => { manager.stop(); process.exit(0); });
    process.on('SIGINT',  () => { manager.stop(); process.exit(0); });

    await manager.init();

    console.log('[QuantileNet] Waiting 10s for Python learner to start...');
    await new Promise(r => setTimeout(r, 10000));

    await manager.runAll();

    console.log('[QuantileNet] All actors finished — data fully sent to learner.');
    process.exit(0);
}

main().catch(err => { console.error('[QuantileNet] Fatal:', err); process.exit(1); });
