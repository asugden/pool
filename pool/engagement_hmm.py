import numpy as np

class EngagementHMM:
    def __init__(self):
        """
        Initialize by iterating through all training runs and getting a sequence for attention
        """

        # NOTE: DO NOT CHANGE HERE.
        # Predictions have to be made for each type of CS
        # and emissions depend on this ordering.
        self.cses = ['plus', 'neutral', 'minus']

        self.attention = None
        self.emissions = None
        self.tprobs = None
        self.cond = None
        self.licks = None
        self.testlicks = None
        self.breaks = None
        self.run_number = []

    def set_runs(self, runs):
        """
        Determine the licking and stimulus sequence data from runs
        :param runs: RunSorter or Run
        :return: self
        """

        if not hasattr(runs, '__len__'):
            runs = [runs]

        self.cond = []
        self.licks = []
        self.testlicks = []
        self.breaks = [0]
        for run in runs:
            self.run_number.append(run.run)

            t2p = run.trace2p()
            conditions, codes = t2p.conditions()
            errors = t2p.errors()
            codes = t2p.codes

            # Correct
            rewlicks = t2p.stimlicks('', 2, 4)
            if 'pavlovian' in codes:
                errors[(conditions == codes['pavlovian']) & (rewlicks > 0)] = 0
                conditions[conditions == codes['pavlovian']] = codes['plus']

            errors[conditions == codes['plus']] = 1 - errors[conditions == codes['plus']]
            conditions[conditions == 9] = -3

            for code in codes:
                if code not in self.cses:
                    conditions[conditions == codes[code]] = -3

            for i, cs in enumerate(self.cses):
                conditions[conditions == codes[cs]] = -i

            self.cond.extend([abs(v) for v in conditions])
            self.licks.extend(errors)
            self.testlicks.extend(rewlicks)
            self.breaks.append(len(self.cond))

        self.cond = np.array(self.cond)
        self.licks = np.array(self.licks)
        self.testlicks = np.array(self.testlicks)

        return self

    def calculate(self):
        """
        Run HMM
        :return: self
        """

        # Initialize HMM
        self.attention = np.zeros(len(self.licks))
        self.init_tprobs()
        self.init_emissions()

        # Run HMM
        self.attention = self.viterbi()

        return self

    def engagement(self, run=None):
        """
        Return the output engagement vector per stimulus
        :param run: Run instance or integer run number
            If None, return engagement across runs
        :return: boolean engagement vector
        """

        if run is None:
            return self.attention < 1

        att = None
        if not isinstance(run, int):
            run = run.run

        for r, runno in enumerate(self.run_number):
            if runno == run:
                att = self.attention[self.breaks[r]:self.breaks[r + 1]] < 1

        return att

    # =====================================================================
    # LOCAL FUNCTIONS

    def viterbi(self):
        """
        Run the viterbi algorithm on the sequence
        :return:
        """

        sev = np.ones(2)  # max probabilities for start and end
        sep = np.zeros(2)  # pointers

        v = np.zeros((2, len(self.cond)))  # start and end come from before
        p = np.zeros((2, len(self.cond)))  # save pointers

        for t in range(len(self.cond)):  # iterate over trials
            for s in range(2):  # iterate over states
                emis = self.emissions[s, self.cond[t]]
                if not self.licks[t]:
                    emis = 1.0 - emis

                if t == 0:
                    v[s, t] = sev[0]*self.tprobs[0, s + 1]*emis
                    p[s, t] = -1

                else:
                    a = v[0, t - 1]*self.tprobs[1, s + 1]*emis
                    b = v[1, t - 1]*self.tprobs[2, s + 1]*emis

                    if a > b:
                        v[s, t] = a
                        p[s, t] = 0
                    else:
                        v[s, t] = b
                        p[s, t] = 1

        a = v[0, -1]*self.tprobs[1, 3]
        b = v[1, -1]*self.tprobs[2, 3]

        if a > b:
            sev[1] = a
            sep[1] = 0
        else:
            sev[1] = b
            sep[1] = 1

        return self.backtrace(p, sep)

    def backtrace(self, p, sep):
        """
        Follow the backtrace through the viterbi path
        :param p: pointers to the previous state
        :param sep: pointers from start and end
        :return: vector of states
        """

        out = np.zeros(np.shape(p)[1], dtype=np.int8)
        out[-1] = sep[1]

        for i in range(np.shape(p)[1] - 2, -1, -1):
            out[i] = p[out[i + 1], i + 1]

        return out

    def init_tprobs(self):
        """
        Initialize the transition probabilities
        :return: None
        """

        # start, engaged, disengaged, end
        self.tprobs = np.array([
            [0.00, 0.90, 0.10, 0.00],
            [0.00, 0.97, 0.02, 0.01],
            [0.00, 0.30, 0.69, 0.01],
            [0.00, 0.00, 0.00, 1.00],
        ])

    def init_emissions(self):
        """
        Initialize the emissions probability that a mouse performs correctly for each stimulus
        :return: None
        """

        # plus, neutral, minus (or same as self.cses)
        # self.emissions = np.array([
        #     [np.mean(self.licks[(self.cond == 0) & self.engaged()]),
        #      1.0 - np.mean(self.licks[(self.cond == 1) & self.engaged()]),
        #      1.0 - np.mean(self.licks[(self.cond == 2) & self.engaged()]),
        #      0.02],
        #     [0.02, 0.02, 0.02, 0.02],
        # ])

        self.emissions = np.array([
            [0.80, 0.40, 0.40, 0.02],
            [0.02, 0.02, 0.02, 0.02],
        ])
