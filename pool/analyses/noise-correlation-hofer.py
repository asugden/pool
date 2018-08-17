# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class NoiseCorrelationHofer(object):
    def __init__(self, data):
        self.out = {}
        ocorr = self.analysis('overly-correlated')

        ncs = self.get_stimuli(data)
        nolick = self.get_stimuli(data, True)

        if not isinstance(ncs, float):
            ncs[ocorr] = np.nan
        if not isinstance(ncs, float):
            nolick[ocorr] = np.nan

        self.out['noise-correlation-hofer'] = ncs
        self.out['noise-correlation-nolick-hofer'] = nolick


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['noise-correlation-hofer', 'noise-correlation-nolick-hofer']
    across = 'day'
    updated = '180222'

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

    def get_stimuli(self, data, nolick=False):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        argsf = {
            'start-s': 0,
            'end-s': 2 if not nolick else 1,
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1 if not nolick else 100,
            'error-trials': -1,
            'baseline': (-1, 0),
        }

        val = {}

        meansubtrs = None
        # Go through the added stimuli and add all onsets
        for cs in ['plus', 'neutral', 'minus']:
            trs = []
            for r in data['train']:
                argsb = deepcopy(argsf)

                if len(trs) == 0:
                    trs = np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)], axis=1)

                if cs == 'plus':
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces('pavlovian', argsb), axis=1)], axis=1)

            trs = (trs.T - np.nanmean(trs, axis=1)).T

            if meansubtrs is None:
                meansubtrs = trs
            else:
                meansubtrs = np.concatenate([meansubtrs, trs], axis=1)

        if not nolick:
            return np.corrcoef(meansubtrs)
        else:
            dftrs = pd.DataFrame(meansubtrs.T)
            return dftrs.corr().as_matrix()
