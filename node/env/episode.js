/**
 * Zarządzanie epizodami tradingowymi.
 * Epizod to sekwencja kroków od losowego punktu startowego
 * do końca danych treningowych lub wymuszonego zakończenia.
 */

class Episode {
    constructor(config) {
        this.trainDataFraction = config.training.train_data_fraction;
        this.maxEpisodeLength  = config.training.episode_length;
        this.stepInterval      = config.training.step_interval || 1;
        this.gamma = config.training.gamma;

        this.steps = [];
        this.startIndex = 0;
        this.currentIndex = 0;
        this.done = false;
    }

    getRandomStartIndex(dataLength) {
        const trainEnd = Math.floor(dataLength * this.trainDataFraction);
        const maxStart = trainEnd - this.maxEpisodeLength * this.stepInterval;
        if (maxStart <= 0) return 0;
        return Math.floor(Math.random() * maxStart);
    }

    start(dataLength) {
        this.steps = [];
        this.done = false;
        this.startIndex = this.getRandomStartIndex(dataLength);
        this.currentIndex = this.startIndex;
        return this.startIndex;
    }

    addStep(state, action, reward, nextState, done, actionMask) {
        this.steps.push({ state, action, reward, nextState, done, actionMask });
        this.currentIndex += this.stepInterval;
        if (done) this.done = true;
    }

    isAtTrainEnd(dataLength) {
        const trainEnd = Math.floor(dataLength * this.trainDataFraction);
        const epEnd    = this.startIndex + this.maxEpisodeLength * this.stepInterval;
        return this.currentIndex >= Math.min(trainEnd, epEnd) - 1;
    }

    getStepCount() {
        return this.steps.length;
    }

    /**
     * Oblicz Monte Carlo Returns idąc od tyłu.
     * G_t = r_t + gamma * G_{t+1}
     */
    calculateMonteCarloReturns() {
        let G = 0;
        for (let t = this.steps.length - 1; t >= 0; t--) {
            G = this.steps[t].reward + this.gamma * G;
            this.steps[t].returnG = G;
        }
    }

    getExperiences() {
        this.calculateMonteCarloReturns();
        return this.steps.map(step => ({
            state: step.state,
            action: step.action,
            reward: step.reward,
            nextState: step.nextState,
            done: step.done,
            actionMask: step.actionMask,
            returnG: step.returnG
        }));
    }

    reset() {
        this.steps = [];
        this.startIndex = 0;
        this.currentIndex = 0;
        this.done = false;
    }
}

module.exports = { Episode };
