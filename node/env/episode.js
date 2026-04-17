/**
 * Zarządzanie epizodami tradingowymi.
 * Epizod to sekwencja kroków od losowego punktu startowego
 * do końca danych treningowych lub wymuszonego zakończenia.
 */

class Episode {
    constructor(config) {
        // validation_weeks ma pierwszeństwo nad validation_days
        this.validationDays   = config.training.validation_weeks
            ? config.training.validation_weeks * 7
            : config.training.validation_days;
        this.maxEpisodeLength = config.training.episode_length;
        this.stepInterval     = config.training.step_interval || 1;
        this.gamma = config.training.gamma;

        // Minimalna pozycja startowa epizodu: sieć potrzebuje candles_1d pełnych świec dziennych
        // jako kontekst. Przy training_months > 0 dane są obcinane od właśnie tego punktu.
        this.warmupCandles = (config.timeframes.candles_1d || 0) * 24 * 60;

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
        // Epizod nie może startować wcześniej niż po warmup — sieć nie miałaby
        // pełnego okna 1d. Przy training_months=0 warmup pochodzi z początku pełnej historii.
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
        // Policz MC returns z góry — tylko do wyświetlania w tabeli epizodu
        this.calculateMonteCarloReturns();

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
                returnG:        this.steps[t].returnG,  // MC return — tylko do wyświetlania
            });
        }
        return result;
    }

    /**
     * Zwraca surowe doświadczenia (1-krokowe TD) bez żadnych transformacji.
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
            returnG:    step.returnG,  // MC return — tylko do wyświetlania
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
