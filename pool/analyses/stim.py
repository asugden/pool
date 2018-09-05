# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
import warnings

from . import base
from pool import config


class Stim(base.AnalysisBase):
    def run(self, mouse, date, training, running, sated, hungry):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        mouse : str
            mouse name
        date : str
            current date
        training : list of ints
            list of training run numbers as integers
        running : list of ints
            list of running-only run numbers as integers
        sated : list of ints
            list of sated spontaneous run numbers as integers
        hungry : list of ints
            list of hungry spontaneous run numbers as integers

        Returns
        -------
        dict
            All of the output values

        """

        out = {}

        for cs in config.stimuli():
            out['stim_dff_%s' % cs] = self.get_stimuli(training, cs, (0, 2), 'dff', False)
            out['stim_dff_alltrials_%s' % cs] = self.get_stimuli(training, cs, (0, 2), 'dff', True)

        return out


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #   classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['stim_dff_%s' % cs for cs in config.stimuli()] + \
           ['stim_dff_alltrials_%s' % cs for cs in config.stimuli()]
    across = 'day'
    updated = '180831'


    # ================================================================================== #
    # ANYTHING YOU NEED

    def get_stimuli(self, runs, cs, trange, ttype, all=False):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        err = -1 if all else 0
        lick = -1 if all else 100
        bl = (-1, 0) if 'dec' not in ttype else (-1, -1)

        # Go through the added stimuli and add all onsets
        trs = []
        for r in runs:
            with warnings.catch_warnings():
                cstrs = self.trace2p(r).cstraces(cs, trange[0], trange[1],
                                                 ttype, lick, err, bl)

                warnings.simplefilter('ignore', category=RuntimeWarning)

                if len(trs) == 0:
                    trs = np.nanmean(cstrs, axis=1)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, np.nanmean(cstrs, axis=1)], axis=1)

                if trange[1] <= 2 and cs == 'plus':
                    pav = self.trace2p(r).cstraces('pavlovian', trange[0], trange[1],
                                                   ttype, lick, err, bl)
                    trs = np.concatenate([trs, np.nanmean(pav, axis=1)], axis=1)

        return np.nanmean(trs, axis=1)