# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

class EQDistDay(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            for cmin in [0.1, 0.2, 0.5]:
                self.out['day-sum-eqdist-%.1f-%s' % (cmin, cs)] = np.nan
                self.out['day-frac-eqdist-%.1f-%s' % (cmin, cs)] = np.nan

        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['day-sum-eqdist-%.1f-%s' % (cmin, cs),
             'day-frac-eqdist-%.1f-%s' % (cmin, cs)]
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]]
    across = 'day'
    updated = '180122'

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

    def geteq(self, data, repmin, cs, group='sated'):
        """
        Get the eq distance
        :param data: data dict, passed to analysis
        :param repmin: replay minimum threshold
        :param cs: stimulus type, plus neutral or minus
        :param group: hungry or sated
        :return: replay fraction with eqdist lo, hi
        """

        allnan = True
        eqds = []

        for run in data[group]:
            reqd = self.analysis('eqdist-run%i-%.1f-%s' % (run, repmin, cs))

            if reqd is not None:
                reqd = np.array(reqd)
                reqd = reqd[np.isfinite(reqd)].tolist()

                if len(reqd) > 0:
                    allnan = False
                    eqds.extend(reqd)

        if not allnan and len(eqds) > 0:
            return np.sum(eqds), float(np.sum(np.array(eqds) > 0))/len(eqds)
        else:
            return np.nan, np.nan

    def analyze(self, data):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        :param pars:
        :param gm:
        :param t2p:
        :param cs:
        :return:
        """

        for cs in ['plus', 'neutral', 'minus']:
            for cmin in [0.1, 0.2, 0.5]:
                rsum, rfrac = self.geteq(data, cmin, cs, 'sated')

                self.out['day-sum-eqdist-%.1f-%s' % (cmin, cs)] = rsum
                self.out['day-frac-eqdist-%.1f-%s' % (cmin, cs)] = rfrac
