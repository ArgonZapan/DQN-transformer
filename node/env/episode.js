/**
 * Trading episode management.
 * An episode is a sequence of steps from a random start index
 * to the end of training data or a forced termination.
 */

class Episode {
    constructor(config) {
        // validation_weeks takes priority over validation_days
        this.validationDays   = config.training.validation_weeks
            ? config.training.validation_weeks * 7
            : config.training.validation_days;
        this.maxEpisodeLength = config.training.episode_length;
        this.stepInterval     = config.training.step_interval || 1;
        this.gamma = config.training.gamma;

        // Minimum episode start position: network needs candles_1d full daily candles
        // as context. With training_months > 0, data is trimmed from this point onward.
        this.warmupCandles = (config.timeframes.candles_1d || 0) * 24 * 60;

        this.steps = [];
        this.startIndex = 0;
        this.currentIndex = 0;
        this.done = false;
    }

    _trainEnd(dataLength) {
        // dataLength = number of 1m candles; last validationDays * 1440 are reserved for validation
        return Math.max(0, dataLength - this.validationDays * 24 * 60);
    }

    getRandomStartIndex(dataLength) {
        const trainEnd = this._trainEnd(dataLength);
        // Episode cannot start before warmup — network wouldn't have a full 1d window.
        // With training_months=0, warmup comes from the beginning of the full history.
        const minStart = this.warmupCandles;
        const maxStart = trainEnd - this.maxEpisodeLength * this.stepInterval;
        if (maxStart <= minStart) return minStart;
        return minStart + Math.floor(Math.random() * (maxStart - minStart));
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
        const trainEnd = this._trainEnd(dataLength);
        const epEnd    = this.startIndex + this.maxEpisodeLength * this.stepInterval;
        return this.currentIndex >= Math.min(trainEnd, epEnd) - 1;
    }

    getStepCount() {
        return this.steps.length;
    }

    /**
     * Compute Monte Carlo returns backwards.
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

    /**
     * Returns experiences with n-step returns.
     *
     * For each step t:
     *   reward    = Σ_{k=0}^{min(n,T-t)-1} γ^k * r_{t+k}   (discounted n-step sum)
     *   nextState = steps[t+n].state  (if t+n < T), otherwise null
     *   done      = (t + n >= T)      (true when episode ends within n steps)
     *
     * Bellman target in trainer:  reward + (1 - done) * γ^n * Q(nextState)
     * γ^n is passed as the `gammaToN` field — trainer multiplies next_q by it.
     */
    getExperiencesNStep(n) {
        // Compute MC returns upfront — used only for episode table display
        this.calculateMonteCarloReturns();

        const T = this.steps.length;
        const gamma = this.gamma;
        const gammaN = Math.pow(gamma, n);
        const result = [];

        for (let t = 0; t < T; t++) {
            // Accumulated n-step discounted return
            let reward = 0;
            for (let k = 0; k < n && (t + k) < T; k++) {
                reward += Math.pow(gamma, k) * this.steps[t + k].reward;
            }

            const endIdx = t + n;
            const done = endIdx >= T;
            const nextState = done ? null : this.steps[endIdx].state;
            const nextActionMask = done ? null : this.steps[endIdx].actionMask;

            result.push({
                state:          this.steps[t].state,
                action:         this.steps[t].action,
                reward,
                nextState,
                done,
                actionMask:     this.steps[t].actionMask,
                nextActionMask,
                gammaToN:       gammaN,
                returnG:        this.steps[t].returnG,  // MC return — display only
            });
        }
        return result;
    }

    /**
     * Returns raw 1-step TD experiences without any transformations.
     */
    getExperiencesTD() {
        this.calculateMonteCarloReturns();
        return this.steps.map(step => ({
            state:      step.state,
            action:     step.action,
            reward:     step.reward,
            nextState:  step.nextState,
            done:       step.done,
            actionMask: step.actionMask,
            gammaToN:   this.gamma,
            returnG:    step.returnG,  // MC return — display only
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
