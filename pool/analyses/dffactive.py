# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
from itertools import chain
import numpy as np
import warnings

class DFFActive(object):
    def __init__(self, data):
        self.out = {}

        for cs in ['plus', 'neutral', 'minus']:
            stimresp = self.analysis('stimulus-dff-all-0-2-%s' % cs)
            self.out['dffactive-%s'%cs] = np.sum(stimresp > 0.025)/float(len(stimresp))
            self.out['good-%s'%cs] = self.out['dffactive-%s'%cs] > 0.05

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['dffactive-%s' % cs, 'good-%s' % cs] for cs in
            ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180117'
    depends_on = ['stimulus-dff-all-0-2-%s' % cs for cs in
                  ['plus', 'neutral', 'minus']]

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
