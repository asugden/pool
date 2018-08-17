# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
import warnings

class StimulusChange(object):
    def __init__(self, data):
        self.out = {}
        self.get_change(data, 'dff')
        self.get_change(data, 'decon')

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #   classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['stimulus-change-%s-%s'%(ttype, cs) for cs in ['plus', 'neutral', 'minus'] for ttype in ['dff', 'decon']]
    across = 'day'
    updated = '170612'

    # def trace2p(self, run):
    #   """
    #   Return trace2p file, automatically injected
    #   :param run: run number, int
    #   :return: trace2p instance
    #   """

    # def classifier(self, run, randomize=''):
    #   """
    #   Return classifier (forced to be created if it doesn't exist), automatically injected
    #   :param run: run number, int
    #   :param randomize: randomization type, optional
    #   :return:
    #   """

    # pars = {}  # dict of parameters, automatically injected

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def get_change(self, data, ttype):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        # Account for weird training days
        if len(data['train']) < 2:
            for cs in ['plus', 'neutral', 'minus']:
                self.out['stimulus-change-%s-%s'%(ttype, cs)] = None

        # Set up the dict to be passed to cstraces
        argsf = {
            'start-s': 0,
            'end-s': 1,
            'trace-type': ttype,
            'cutoff-before-lick-ms': 100,  # only up to 100 ms before the first lick
            'error-trials': 0,  # only correct trials
            'baseline': (-1, 0),
        }

        # Go through the added stimuli and add all onsets
        for cs in ['plus', 'neutral', 'minus']:
            rmn = min(data['train'])
            rmx = max(data['train'])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                trs = np.nanmean(self.trace2p(rmn).cstraces(cs, argsf), axis=1)  # ncells, frames, nstimuli/onsets
                trsx = np.nanmean(self.trace2p(rmx).cstraces(cs, argsf), axis=1)  # ncells, frames, nstimuli/onsets

                if cs == 'plus':
                    pav = self.trace2p(rmn).cstraces('pavlovian', argsf)
                    trs = np.concatenate([trs, np.nanmean(pav, axis=1)], axis=1)

                    pavx = self.trace2p(rmx).cstraces('pavlovian', argsf)
                    trsx = np.concatenate([trsx, np.nanmean(pavx, axis=1)], axis=1)

            self.out['stimulus-change-%s-%s'%(ttype, cs)] = np.nanmean(trsx, axis=1) - np.nanmean(trs, axis=1)