# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
from scipy.stats import norm

class Behavior(object):
    def __init__(self, data):
        self.ncells = -1
        self.out = {}
        self.activity(data)
        self.activity_spontaneous(data, 'sated')
        self.activity_spontaneous(data, 'hungry')

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['activity%s-mean'%t, 'activity%s-median'%t, 'activity%s-deviation'%t, 'activity%s-outliers'%t] for t in
            ['', '-sated', '-hungry']]
    across = 'day'
    updated = '180116'

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

    def activity(self, data):
        """
        Get the mean population activity level and variance, based on non-stimulus periods
        :param data:
        :return:
        """
        popact = []
        for r in data['train']:
            t2p = self.trace2p(r)
            self.ncells = t2p.ncells
            pact = np.nanmean(t2p.trace('deconvolved'), axis=0)
            skipframes = int(t2p.framerate*4)

            for cs in ['plus', 'neutral', 'minus', 'pavlovian']:
                onsets = t2p.csonsets(cs)
                for ons in onsets:
                    pact[ons:ons+skipframes] = np.nan
            popact = np.concatenate([popact, pact[np.isfinite(pact)]])

        self.out['activity-median'] = np.median(popact)

        popact = self.exclude_extremes(popact)

        self.out['activity-mean'] = np.mean(popact)
        self.out['activity-deviation'] = np.std(popact)
        self.out['activity-outliers'] = np.array([False for i in range(self.ncells)])

    def exclude_extremes(self, data):
        """
        Remove the top and bottom 10th percent
        :param data: pop activity vector
        :return: data vector with extremes removed
        """

        percent = 2.0
        data = np.sort(data)
        trim = int(percent*data.size/100.0)
        return data[trim:-trim]

    def activity_spontaneous(self, data, atype):
        """
        Get the mean population activity level and variance, based on non-stimulus periods
        :param data:
        :return:
        """

        self.out['activity-%s-mean'%atype] = None
        self.out['activity-%s-median'%atype] = None
        self.out['activity-%s-deviation'%atype] = None
        self.out['activity-%s-outliers'%atype] = np.array([False for i in range(self.ncells)])

        if atype not in data or len(data[atype]) == 0: return

        popact = []
        outliers = []
        for r in data[atype]:
            t2p = self.trace2p(r)
            pact = t2p.trace('deconvolved')
            fmin = t2p.lastonset()
            mask = t2p.inactivity()
            mask[:fmin] = False

            # pact[:, np.invert(mask)] = np.nan
            if len(popact) == 0:
                popact = pact[:, mask]
            else:
                popact = np.concatenate([popact, pact[:, mask]], axis=1)

            trs = t2p.trace('deconvolved')[:, fmin:]
            cellact = np.nanmean(trs, axis=1)
            outs = cellact > np.nanmedian(cellact) + 2*np.std(cellact)

            if len(outliers) == 0:
                outliers = outs
            else:
                outliers = np.bitwise_or(outliers, outs)

        popact = np.nanmean(popact[np.invert(outliers), :], axis=0)

        if len(popact) > 0:
            self.out['activity-%s-median'%atype] = np.median(popact)
            # popact = self.exclude_extremes(popact)
            self.out['activity-%s-mean'%atype] = np.mean(popact)
            self.out['activity-%s-deviation'%atype] = np.std(popact)
            self.out['activity-%s-outliers'%atype] = outliers
