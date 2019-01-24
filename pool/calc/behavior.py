"""Behavior analyses."""
import numpy as np
# from scipy.stats import norm

from ..database import memoize


@memoize(across='date', updated=190124)
def engaged(date):
    hmm = HMM()
    hmm.getseqs([run.trace2p() for run in date.runs('training')])
    hmm.run()

    return hmm.engaged()

class HMM:
    def __init__(self):
        """
        Initialize by iterating through all training runs and getting a sequence for attention
        """

        self.cses = ['plus', 'neutral', 'minus']

        self.attention = None
        self.emissions = None
        self.tprobs = None
        self.cond = None
        self.licks = None
        self.testlicks = None

    def run(self):
        """
        Run HMM
        :return: vals
        """

        # Initialize HMM
        self.attention = np.zeros(len(self.licks))
        self.init_tprobs()
        self.init_emissions()

        # Run HMM
        self.attention = self.viterbi()

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
                    v[s, t] = sev[0]*self.tprobs[0, s+1]*emis
                    p[s, t] = -1

                else:
                    a = v[0, t - 1]*self.tprobs[1, s+1]*emis
                    b = v[1, t - 1]*self.tprobs[2, s+1]*emis

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
            out[i] = p[out[i+1], i+1]

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
        self.emissions = np.array([
            [0.80, 0.40, 0.40, 0.02],
            [0.02, 0.02, 0.02, 0.02],
        ])

    # Local functions
    def getseqs(self, t2ps):
        """
        Get sequence of stimulus types
        :param t2ps: list of trace2p instances
        :return: None
        """

        self.cond = []
        self.licks = []
        self.testlicks = []
        self.breaks = [0]
        for t2p in t2ps:
            condd, errd, coded = t2p.conderrs()

            # Correct
            condd = condd.astype(np.int16)
            rewlicks = t2p.stimlicks('', 2, 4)
            if 'pavlovian' in coded:
                errd[(condd == coded['pavlovian']) & (rewlicks > 0)] = 0
                condd[condd == coded['pavlovian']] = coded['plus']

            errd[condd == coded['plus']] = 1 - errd[condd == coded['plus']]
            condd[condd == 9] = -3

            for code in coded:
                if code not in self.cses:
                    condd[condd == coded[code]] = -3

            for i, cs in enumerate(self.cses):
                condd[condd == coded[cs]] = -i

            self.cond.extend([abs(v) for v in condd])
            self.licks.extend(errd)
            self.testlicks.extend(rewlicks)
            self.breaks.append(len(self.cond))

        self.cond = np.array(self.cond)
        self.licks = np.array(self.licks)
        self.testlicks = np.array(self.testlicks)

    # def setruns(self):
    #     """
    #     Using "breaks", set individual runs
    #     :return:
    #     """

    #     dprimes = [np.nan, np.nan, np.nan]
    #     criteria = [np.nan, np.nan, np.nan]

    #     for r in range(min(3, len(self.breaks) - 1)):
    #         lick = self.licks[self.breaks[r]:self.breaks[r+1]]
    #         condition = self.cond[self.breaks[r]:self.breaks[r+1]]
    #         att = self.attention[self.breaks[r]:self.breaks[r+1]]
    #         dprimes[r], criteria[r], _ = self.calc_sdt_metrics(lick, condition, att)

    #     return np.concatenate([dprimes, criteria])

    # def setrunengagement(self):
    #     """
    #     Using "breaks", set individual runs.

    #     :return:

    #     """
    #     eng = [[], [], []]

    #     for r in range(min(3, len(self.breaks) - 1)):
    #         eng[r] = self.attention[self.breaks[r]:self.breaks[r+1]] < 1

    #     return eng

    def engaged(self):
        """
        Return the output engagement vector per stimulus.

        :return: boolean engagement vector

        """
        return self.attention < 1

    # def behavior(self, cs=''):
    #     """
    #     Return the behavior plus, neutral, or minus
    #     :param cs: optional parameter for stimulus, if empty then average of three cses
    #     :return: percent behavior correct
    #     """

    #     if np.sum(self.attention == 0) == 0:
    #         return np.nan

    #     translation = {'plus': 0, 'neutral': 1, 'minus': 2}

    #     if len(cs) > 0:
    #         masked_behavior = \
    #             (self.cond == translation[cs]) & self.engaged()
    #         if np.sum(masked_behavior) == 0:
    #             return np.nan
    #         else:
    #             result = np.mean(self.licks[masked_behavior])
    #             if cs in ['minus', 'neutral']:
    #                 result = 1.0 - result
    #             return result
    #     else:
    #         nhits = np.sum(
    #             self.licks[(self.cond == 0) & self.engaged()])
    #         nplus = len(
    #             self.licks[(self.cond == 0) & self.engaged()])
    #         nfas = np.sum(
    #             self.licks[(self.cond == 2) & self.engaged()])
    #         npassives = len(
    #             self.licks[(self.cond == 2) & self.engaged()])
    #         nfas += np.sum(
    #             self.licks[(self.cond == 1) & self.engaged()])
    #         npassives += len(
    #             self.licks[(self.cond == 1) & self.engaged()])

    #         if nplus + npassives == 0:
    #             return np.nan
    #         else:
    #             return float(nhits + (npassives - nfas)) / (nplus + npassives)

    # def behavior_counts(self, cs):
    #     """
    #     Return the number of correct/wrong trials for plus, neutral, or minus
    #     :param cs: parameter for stimulus
    #     :return: (ncorrect, nwrong)
    #     """

    #     if np.sum(self.attention == 0) == 0:
    #         return np.nan, np.nan

    #     translation = {'plus': 0, 'neutral': 1, 'minus': 2}

    #     if np.sum((self.cond == translation[cs]) & self.engaged()) == 0:
    #         return np.nan, np.nan
    #     else:
    #         n_lick = np.sum(
    #             self.licks[(self.cond == translation[cs]) &
    #                        self.engaged()])
    #         n_no_lick = np.sum(
    #             self.licks[(self.cond == translation[cs]) &
    #                        self.engaged()] == 0)
    #         if cs == 'plus':
    #             return n_lick, n_no_lick
    #         elif cs in ['neutral', 'minus']:
    #             return n_no_lick, n_lick
    #         else:
    #             raise ValueError('Unrecognized cs type: {}'.format(cs))

    # def sdt_metrics(self, signal='p', noise='mn'):
    #     return self.calc_sdt_metrics(self.licks, self.cond, self.attention, signal=signal, noise=noise)


    # def calc_sdt_metrics(self, licks, condition, attention, signal='p', noise='mn'):
    #     """Calculate signal detection theory behavioral metrics.

    #     :return: (dprime, criterion, LR)

    #     """
        
    #     translation = {'p': 0, 'n': 1, 'm': 2}

    #     nhits = 0
    #     nsignal_trials = 0
    #     for trial_type in signal:
    #         trial_licks = licks[
    #             (condition == translation[trial_type]) & (attention < 1)]

    #         nhits += np.sum(trial_licks)
    #         nsignal_trials += len(trial_licks)

    #     nfas = 0
    #     nnoise_trials = 0
    #     for trial_type in noise:
    #         trial_licks = licks[
    #             (condition == translation[trial_type]) & (attention < 1)]

    #         nfas += np.sum(trial_licks)
    #         nnoise_trials += len(trial_licks)

    #     z_hit_rate = norm.ppf((nhits + 0.5) / (nsignal_trials + 1.0))
    #     z_fa_rate = norm.ppf((nfas + 0.5) / (nnoise_trials + 1.0))

    #     dprime = z_hit_rate - z_fa_rate
    #     c = -1 * (z_hit_rate + z_fa_rate) / 2.
    #     lr = np.exp((z_fa_rate**2 - z_hit_rate**2) / 2.)

    #     return dprime, c, lr

    # def getchange(self, out):
    #     """
    #     Get the change within day
    #     :return: None
    #     """

    #     mxrun = 4
    #     if not np.isfinite(out['hmm-dprime-run4']):
    #         mxrun -= 1
    #         if not np.isfinite(out['hmm-dprime-run3']):
    #             mxrun -= 1

    #     if mxrun == 2:
    #         out['hmm-delta-dprime'] = np.nan
    #         out['hmm-delta-criterion'] = np.nan
    #     else:
    #         out['hmm-delta-dprime'] = out['hmm-dprime-run%i' % mxrun] - out['hmm-dprime-run2']
    #         out['hmm-delta-criterion'] = out['hmm-criterion-run%i' % mxrun] - out['hmm-criterion-run2']

    #     return out
