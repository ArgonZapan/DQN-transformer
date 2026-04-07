/**
 * Normalizacja danych per para kryptowalutowa.
 * Każda para ma własne statystyki rolling (mean, std).
 */

class PerPairNormalizer {
    constructor(window) {
        this.window = window;
        this.history = [];
    }

    update(close) {
        this.history.push(close);
        if (this.history.length > this.window) {
            this.history.shift();
        }
    }

    getMean() {
        if (this.history.length === 0) return 0;
        return this.history.reduce((a, b) => a + b, 0) / this.history.length;
    }

    getStd() {
        if (this.history.length < 2) return 0;
        const mean = this.getMean();
        const variance = this.history.reduce((sum, v) => sum + (v - mean) ** 2, 0) / this.history.length;
        return Math.sqrt(variance);
    }

    normalize(value) {
        if (this.history.length < 2) return 0;

        const mean = this.getMean();
        const std = this.getStd();

        if (std === 0) return 0;
        return (value - mean) / std;
    }

    getState() {
        return { window: this.window, history: [...this.history] };
    }

    loadState(state) {
        this.window = state.window;
        this.history = [...state.history];
    }
}

module.exports = { PerPairNormalizer };
