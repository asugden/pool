# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr

class SpontaneousCorrelation(object):
    def __init__(self, data):
        self.out = {}
        self.get_stimuli(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['spontaneous-correlation', 'overly-correlated']
    across = 'day'
    updated = '180218'

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

    def fixnans(self, trs):
        """
        Remove NANs
        :param trs:
        :return:
        """

        ncells = np.shape(trs)[0]
        ntimes = np.shape(trs)[1]

        badcells = np.zeros(ncells)
        for c in range(len(badcells)):
            if np.sum(np.isnan(trs[c, :])) > 0.2*ntimes:
                badcells[c] = 1

        goodcells = badcells < 1

        badtimes = np.sum(np.isnan(trs[goodcells, :]), axis=0)
        return trs[:, badtimes < 1]

    def get_stimuli(self, data):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        pars = deepcopy(self.pars)

        # Go through the added stimuli and add all onsets
        trs = []
        for r in data['train']:
            t2p = self.trace2p(r)
            act = t2p.trace('deconvolved')
            ons = t2p.csonsets('')
            win = [int(round(-t2p.framerate*0.5)), int(round(t2p.framerate*4))]
            badt = np.zeros(np.shape(act)[1])
            for onset in ons:
                badt[max(0, onset+win[0]):onset+win[1]] = 1

            if len(trs) == 0:
                trs = act[:, badt < 1]
            else:
                trs = np.concatenate([trs, act[:, badt < 1]], axis=1)

        trs = self.fixnans(trs)
        self.out['spontaneous-correlation'] = np.corrcoef(trs)

        # nrand = 100
        # ncells = np.shape(trs)[0]
        # stimorder = np.arange(np.shape(trs)[1])
        # for i in range(nrand):
        #     for c in range(ncells):
        #         np.random.shuffle(stimorder)
        #         trs[c, :] = trs[c, stimorder]
        #     self.out['spontaneous-correlation'] -= np.corrcoef(trs)/float(nrand)

        # Mean across all cells is 0.0038 +- 0.0188
        # Exclude those with values 5 sigma above the mean

        nanpos = np.isnan(self.out['spontaneous-correlation'])
        self.out['spontaneous-correlation'][nanpos] = 0
        self.out['overly-correlated'] = self.out['spontaneous-correlation'] > 0.0038 + 8*0.0188
        self.out['spontaneous-correlation'][self.out['overly-correlated']] = np.nan
        self.out['spontaneous-correlation'][nanpos] = np.nan