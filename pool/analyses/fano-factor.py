# Updated: 170410
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
from scipy.stats import norm

class FanoFactor(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['fano-factor-%s'%cs, 'fano-factor-decon-%s'%cs] for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180126'

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

    def ff(self, data, cs, tracetype='dff'):
        """
        Get the fano factor for a particular trace type
        :param data: passed data to the class-- dict of days
        :param cs: stimulus str, 'plus', 'neutral', or 'minus'
        :param tracetype: str, 'dff' or 'deconvolved'
        :return: fano factor vector
        """

        args = {
            'start-s': 0,
            'end-s': 2,
            'trace-type': tracetype,
            'cutoff-before-lick-ms': -1,
            'baseline': (-1, 0),
            'error-trials': 0,
        }

        vals = []
        for r in data['train']:
            t2p = self.trace2p(r)

            # ncells, frames, nstimuli/onsets
            trs = t2p.cstraces(cs, args)
            if cs == 'plus':
                pavs = t2p.cstraces('pavlovian', args)
                trs = np.concatenate([trs, pavs], axis=2)

            if len(vals) == 0:
                vals = trs
            else:
                vals = np.concatenate([vals, trs], axis=2)

        if len(vals) == 0: return None
        vals = np.nanmean(vals, axis=1)
        mnresponse = np.nanmean(vals, axis=1)
        stdev = np.std(vals, axis=1)
        mnresponse[mnresponse == 0] = np.nan

        return np.abs(stdev*stdev/mnresponse)

    def analyze(self, data):
        """
        Get the fano factor for each cell
        :param data:
        :return: None
        """

        for cs in ['plus', 'neutral', 'minus']:
            self.out['fano-factor-%s'%(cs)] = self.ff(data, cs, 'dff')
            self.out['fano-factor-decon-%s'%(cs)] = self.ff(data, cs, 'deconvolved')
