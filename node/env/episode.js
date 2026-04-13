/**
 * Zarządzanie epizodami tradingowymi.
 * Epizod to sekwencja kroków od losowego punktu startowego
 * do końca danych treningowych lub wymuszonego zakończenia.
 */

class Episode {
    constructor(config) {
        this.validationDays   = config.training.validation_days;
        this.maxEpisodeLength = config.training.episode_length;
        this.stepInterval     = config.training.step_interval || 1;
        this.gamma = config.training.gamma;

        this.steps = [];
        this.startIndex = 0;
        this.currentIndex = 0;
        this.done = false;
    }

    _trainEnd(dataLength) {
        // dataLength = liczba świec 1m; ostatnie validationDays * 1440 to walidacja
        return Math.max(0, dataLength - this.validationDays * 24 * 60);
    }

    getRandomStartIndex(dataLength) {
        const trainEnd = this._trainEnd(dataLength);
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
        const trainEnd = this._trainEnd(dataLength);
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

    /**
     * Zwraca doświadczenia z n-krokowym zwrotem.
     *
     * Dla każdego kroku t:
     *   reward  = Σ_{k=0}^{min(n,T-t)-1} γ^k * r_{t+k}   (zdyskontowana suma n kroków)
     *   nextState = steps[t+n].state  (jeśli t+n < T), inaczej null
     *   done    = (t + n >= T)         (True gdy epizod kończy się w obrębie n kroków)
     *
     * Bellman target w trainerze:  reward + (1 - done) * γ^n * Q(nextState)
     * γ^n jest przekazywane jako pole `gammaToN` — trainer mnoży przez nie next_q.
     */
    getExperiencesNStep(n) {
        const T = this.steps.length;
        const gamma = this.gamma;
        const gammaN = Math.pow(gamma, n);
        const result = [];

        for (let t = 0; t < T; t++) {
            // Skumulowany n-krokowy zwrot
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
            });
        }
        return result;
    }

    /**
     * Zwraca surowe doświadczenia (1-krokowe TD) bez żadnych transformacji.
     */
    getExperiencesTD() {
        return this.steps.map(step => ({
            state:      step.state,
            action:     step.action,
            reward:     step.reward,
            nextState:  step.nextState,
            done:       step.done,
            actionMask: step.actionMask,
            gammaToN:   this.gamma,
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
