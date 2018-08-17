# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
from scipy import stats

class VisuallyDriven(object):
    """
    Set the log inverse p values of the visual-drivenness of cells.
    visually-driven-[cs] = -1*log(visual driven p value) where mean responses < 0 are set to 0
    """

    def __init__(self, data):
        self.out = {}
        self.analyze(data)

        mxes = np.array([self.out['visually-driven-plus'], self.out['visually-driven-neutral'],
                         self.out['visually-driven-minus']])
        self.out['visually-driven-hofer'] = np.nanmax(mxes, axis=0)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['visually-driven-%s'%cs,
             'stimulus-drive-%s'%cs,
             'fraction-visually-driven-%s'%cs,
             'fraction-visually-driven-30-%s'%cs,
             'fraction-visually-driven-50-%s'%cs,]
            for cs in ['plus', 'neutral', 'minus']] + \
            ['visually-driven-hofer']
    across = 'day'
    updated = '180314'

    # def trace2p(self, run):
    # 	"""
    # 	Return trace2p file, automatically injected
    # 	:param run: run number, int
    # 	:return: trace2p instance
    # 	"""

    # def classifier(self, run, randomize=''):
    # 	"""
    # 	Return classifier (forced to be created if it doesn't exist), automatically injected
    # 	:param run: run number, int
    # 	:param randomize: randomization type, optional
    # 	:return:
    # 	"""

    # pars = {}  # dict of parameters, automatically injected

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def medianfirstlick(self, data, cs):
        """
        Get the amount of total running and the running per CS
        :param data:
        :return:
        """

        firstlicks = []
        for r in data['train']:
            t2p = self.trace2p(r)
            fl = t2p.firstlick(cs, units='frames', maxframes=t2p.framerate*2)
            fl[np.isnan(fl)] = int(round(t2p.framerate*2))
            if len(firstlicks) == 0:
                firstlicks = fl
            else:
                firstlicks = np.concatenate([firstlicks, fl], axis=0)

        if len(firstlicks) < 2: return int(round(t2p.framerate*2))
        return np.nanmedian(firstlicks)

    def gettrials(self, data, cs, oargs):
        """
        Get trials for all training days using t2p.cstraces
        :param data:
        :param cs:
        :return:
        """

        alltrs = []
        for r in data['train']:
            t2p = self.trace2p(r)

            trs = t2p.cstraces(cs, oargs)  # ncells, frames, nstimuli/onsets
            if cs == 'plus':
                pavs = t2p.cstraces('pavlovian', oargs)
                trs = np.concatenate([trs, pavs], axis=2)

            if len(alltrs) == 0:
                alltrs = trs
            else:
                alltrs = np.concatenate([alltrs, trs], axis=2)

        alltrs = np.nanmean(alltrs, axis=1)  # across frames
        return alltrs

    def baselines(self, data, cs):
        """
        Get the baseline dff responses
        :param data:
        :param cs:
        :return:
        """

        args = {
            'start-s': -1,
            'end-s': 0,
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1,  #
            'error-trials': -1,  # -1 is off, 0 is only correct trials, 1 is error trials
        }

        return self.gettrials(data, cs, args)

    def responses(self, data, cs, tintegrate=0.3, pval=0.05, ncses=3, nolick=False):
        """
        Get the responses integrating across different time windows
        :param data:
        :param cs:
        :param tintegrate: integration time in seconds, will be converted to frames
        :return:
        """

        mfl = self.medianfirstlick(data, cs)
        fr = self.trace2p(data['train'][0]).framerate
        fintegrate = int(round(tintegrate*fr))

        # Cut off the first number after the median first lick
        ts = np.arange(0, 2*fr+1, fintegrate)
        am = np.argmax(ts > mfl)
        if np.max(ts) > mfl and am < len(ts) - 1: ts = ts[:am+1]

        args = {
            'start-s': 0,
            'end-s': 0,
            'trace-type': 'dff',
            'cutoff-before-lick-ms': 100 if not nolick else -1,
            'error-trials': 0,  # only correct trials
        }

        bls = self.baselines(data, cs)
        meanbl = np.nanmean(bls, axis=1)

        vdriven = np.zeros(np.shape(bls)[0], dtype=bool)
        pval /= len(ts) - 1  # Correct for number of time points
        pval /= np.shape(bls)[0]  # Correct for the number of cells
        pval /= ncses  # Correct for number of CSes

        # We will save the maximum inverse p values
        maxinvps = np.zeros(np.shape(bls)[0], dtype=np.float64)

        for i in range(len(ts) - 1):
            targs = deepcopy(args)
            targs['start-s'] = float(ts[i])/fr
            targs['end-s'] = float(ts[i+1])/fr
            trs = self.gettrials(data, cs, targs)

            for c in range(np.shape(trs)[0]):
                if np.nanmean(trs[c, :]) > meanbl[c]:
                    pv = stats.ranksums(bls[c, :], trs[c, :]).pvalue
                    logpv = -1*np.log(stats.ranksums(bls[c, :], trs[c, :]).pvalue)
                    if logpv > maxinvps[c]: maxinvps[c] = logpv
                    if pv <= pval:
                        vdriven[c] = True

        return vdriven, maxinvps

    def stronglydriven(self, pvals, cs):
        """
        Return the strongly driven cells
        :return:
        """

        # Will return the responses for all cells
        out = np.zeros(len(pvals))

        # Select only visually driven cells
        pval_threshold = -1*np.log(0.01/3.0)
        driven = pvals > pval_threshold

        out[driven] = self.analysis('stimulus-dff-0-1-%s'%(cs))[driven]
        return out

    def analyze(self, data):
        csresps = {}
        totdriven80 = 0.0
        totdriven30 = 0.0
        totdriven50 = 0.0
        skipday = False
        for cs in ['plus', 'neutral', 'minus']:
            _, self.out['visually-driven-%s'%cs] = self.responses(data, cs, 0.3, ncses=3, nolick=True)
            totdriven80 += np.nansum(self.out['visually-driven-%s' % cs] > 80)
            totdriven30 += np.nansum(self.out['visually-driven-%s' % cs] > 30)
            totdriven50 += np.nansum(self.out['visually-driven-%s' % cs] > 50)
            self.out['stimulus-drive-%s'%cs] = self.stronglydriven(self.out['visually-driven-%s' % cs], cs)

            if not self.analysis('good-%s' % cs):
                skipday = True

        for cs in ['plus', 'neutral', 'minus']:
            drive80 = np.nan if totdriven80 < 1 else np.nansum(self.out['visually-driven-%s' % cs] > 80)/totdriven80
            drive30 = np.nan if totdriven30 < 1 else np.nansum(self.out['visually-driven-%s' % cs] > 30)/totdriven30
            drive50 = np.nan if totdriven50 < 1 else np.nansum(self.out['visually-driven-%s' % cs] > 50)/totdriven50

            if not skipday:
                self.out['fraction-visually-driven-%s' % cs] = drive80
                self.out['fraction-visually-driven-30-%s' % cs] = drive30
                self.out['fraction-visually-driven-50-%s' % cs] = drive50
            else:
                self.out['fraction-visually-driven-%s' % cs] = np.nan
                self.out['fraction-visually-driven-30-%s' % cs] = np.nan
                self.out['fraction-visually-driven-50-%s' % cs] = np.nan
