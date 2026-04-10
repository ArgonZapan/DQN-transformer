const zmq = require('zeromq');
const msgpack = require('msgpack5')();

class PythonClient {
    constructor(config) {
        this.host = config.learner.host;
        this.port = config.learner.port;
        this.socket = new zmq.Request();
        this._connected = false;
    }

    async connect() {
        const address = `${this.host}:${this.port}`;
        this.socket.connect(address);
        this._connected = true;
        console.log(`[PythonClient] Connected to ${address}`);
    }

    async sendStep(data) {
        const encoded = msgpack.encode(data);
        await this.socket.send(encoded);
        const [reply] = await this.socket.receive();
        return msgpack.decode(reply);
    }
    
    getStats() {
        return {
            connected: this._connected,
            address: `${this.host}:${this.port}`
        };
    }

    async sendBatch(batch) {
        const encoded = msgpack.encode(batch);
        await this.socket.send(encoded);
        const [reply] = await this.socket.receive();
        return msgpack.decode(reply);
    }

    async predict(state, actionMask) {
        const data = { state };
        if (actionMask) data.actionMask = actionMask;
        const encoded = msgpack.encode(data);
        await this.socket.send(encoded);
        const [reply] = await this.socket.receive();
        return msgpack.decode(reply);
    }

    close() {
        if (this._connected) {
            this.socket.close();
            this._connected = false;
            console.log('[PythonClient] Connection closed.');
        }
    }
}

module.exports = { PythonClient };
