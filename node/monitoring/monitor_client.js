const zmq = require('zeromq');

class MonitorClient {
    constructor(config) {
        this.host = config.monitoring.host;
        this.port = config.monitoring.metrics_pull_port;
        this.interval = config.monitoring.metrics_push_interval_sec * 1000;
        this.socket = new zmq.Push();
        this._connected = false;
    }

    async connect() {
        const address = `${this.host}:${this.port}`;
        this.socket.connect(address);
        this._connected = true;
        console.log(`[MonitorClient] Connected to ${address}`);
    }

    async sendMetrics(source, metrics) {
        if (!this._connected) return;

        try {
            const message = JSON.stringify({ source, metrics, timestamp: Date.now() });
            await this.socket.send(message);
        } catch (err) {
            console.warn(`[MonitorClient] Push failed: ${err.message}`);
        }
    }

    close() {
        if (this._connected) {
            this.socket.close();
            this._connected = false;
        }
    }
}

module.exports = { MonitorClient };
