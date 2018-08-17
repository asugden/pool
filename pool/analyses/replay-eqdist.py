# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import events, outfns


class ReplayEQDist(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            for cmin in [0.1, 0.2, 0.5]:
                for group in ['lo', 'lo-hungry', 'hi', 'hi-hungry']:
                    self.out['replay-freq-eqdist%s-%.1f-%s' % (group, cmin, cs)] = np.nan

        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['replay-freq-eqdistlo-%.1f-%s' % (cmin, cs),
             'replay-freq-eqdistlo-hungry-%.1f-%s' % (cmin, cs),
             'replay-freq-eqdisthi-%.1f-%s' % (cmin, cs),
             'replay-freq-eqdisthi-hungry-%.1f-%s' % (cmin, cs)]
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]]
    across = 'day'
    updated = '180129'

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

        if self.analysis('good-%s' % cs):
            hi, lo, ml, fr, tot = 0.0, 0.0, 0.0, 0.0, 0.0
            for run in data[group]:
                ml += self.analysis('mask-length-run%i' % run)
                fr = self.analysis('framerate-run%i' % run)
                eqds = np.array(outfns.emptynone(self.analysis('eqdist-run%i-%.1f-%s' % (run, repmin, cs))))

                hi += np.sum(eqds > 0)
                lo += np.sum(eqds < 0)

            if ml > 0:
                return float(lo)/ml*fr, float(hi)/ml*fr

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
                lo, hi = self.geteq(data, cmin, cs, 'hungry')
                self.out['replay-freq-eqdistlo-hungry-%.1f-%s' % (cmin, cs)] = lo
                self.out['replay-freq-eqdisthi-hungry-%.1f-%s' % (cmin, cs)] = hi

                lo, hi = self.geteq(data, cmin, cs, 'sated')
                self.out['replay-freq-eqdistlo-%.1f-%s' % (cmin, cs)] = lo
                self.out['replay-freq-eqdisthi-%.1f-%s' % (cmin, cs)] = hi
