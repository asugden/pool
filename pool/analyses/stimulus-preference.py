# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
from itertools import chain
import numpy as np
import warnings

class StimulusResponse(object):
    def __init__(self, data):
        self.out = {}

        denom = None
        vals = {}
        for cs in ['plus', 'neutral', 'minus']:
            vals[cs] = self.analysis('stimulus-dff-0-2-%s' % cs)
            vals[cs][vals[cs] < 0] = 0.0

            if denom is None:
                denom = np.copy(vals[cs])
            else:
                denom += vals[cs]

        for cs in ['plus', 'neutral', 'minus']:
            out = np.zeros(len(denom))
            out[denom <= 0] = np.nan
            out[denom > 0] = vals[cs][denom > 0]/denom[denom > 0]
            self.out['stimulus-pref-0-2-%s' % cs] = out

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['stimulus-pref-0-2-%s' % cs for cs in ['plus', 'neutral', 'minus']]
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
