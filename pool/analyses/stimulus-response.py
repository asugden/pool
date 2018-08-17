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

        for cs in ['plus', 'neutral', 'minus', 'disengaged1', 'disengaged2']:
            for t in [(0, 1), (1, 2), (0, 2), (2, 4)]:
                for tracetype in ['dff', 'decon']:
                    self.out['stimulus-%s-%i-%i-%s'%(tracetype, t[0], t[1], cs)] = self.get_stimuli(data, cs, t, tracetype, False)
                    self.out['stimulus-%s-all-%i-%i-%s'%(tracetype, t[0], t[1], cs)] = self.get_stimuli(data, cs, t, tracetype, True)


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #   classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['stimulus-dff-%i-%i-%s'%(t[0], t[1], cs),
                'stimulus-dff-all-%i-%i-%s'%(t[0], t[1], cs),
                'stimulus-decon-%i-%i-%s'%(t[0], t[1], cs),
                'stimulus-decon-all-%i-%i-%s'%(t[0], t[1], cs),
            ] for cs in ['plus', 'neutral', 'minus', 'disengaged1', 'disengaged2'] for t in [(0, 1), (1, 2), (0, 2), (2, 4)]]
    across = 'day'
    updated = '170831'

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

    def get_stimuli(self, data, cs, trange, ttype, all=False):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        argsf = {
            'start-s': trange[0],
            'end-s': trange[1],
            'trace-type': ttype,
            'cutoff-before-lick-ms': -1 if all else 100,
            'error-trials': -1 if all else 0,
            'baseline': (-1, 0) if ttype == 'dff' else (-1, -1),
        }

        # Go through the added stimuli and add all onsets
        trs = []
        for r in data['train']:
            argsb = deepcopy(argsf)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if len(trs) == 0:
                    trs = np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)], axis=1)

                if argsb['end-s'] <= 2 and cs == 'plus':
                    pav = self.trace2p(r).cstraces('pavlovian', argsb)
                    trs = np.concatenate([trs, np.nanmean(pav, axis=1)], axis=1)

        return np.nanmean(trs, axis=1)