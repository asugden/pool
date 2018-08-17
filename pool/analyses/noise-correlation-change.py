# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr

class NoiseCorrelationChange(object):
    def __init__(self, data):
        self.out = {'noise-correlation-change-%s' % cs: np.nan for cs in ['plus', 'neutral', 'minus']}
        self.get_stimuli(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['noise-correlation-change-%s' % cs for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180127'

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

    def get_stimuli(self, data):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        argsf = {
            'start-s': 0,
            'end-s': 2,
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1,
            'error-trials': -1,
            'baseline': (-1, 0),
        }

        # Go through the added stimuli and add all onsets
        for cs in ['plus', 'neutral', 'minus']:
            nc = [[], []]
            for j, r in enumerate([data['train'][0], data['train'][1]]):
                argsb = deepcopy(argsf)

                trs = np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)  # ncells, frames, nstimuli/onsets
                if cs == 'plus':
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces('pavlovian', argsb), axis=1)], axis=1)

                nc[j] = np.corrcoef(trs)

                nrand = 100
                ncells = np.shape(trs)[0]
                stimorder = np.arange(np.shape(trs)[1])
                for i in range(nrand):
                    for c in range(ncells):
                        np.random.shuffle(stimorder)
                        trs[c, :] = trs[c, stimorder]
                    nc[j] -= np.corrcoef(trs)/float(nrand)

            self.out['noise-correlation-change-%s' % cs] = np.array(nc[1]) - np.array(nc[0])
