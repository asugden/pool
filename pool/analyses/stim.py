import numpy as np
import warnings

from . import base
from pool import config


class Stim(base.AnalysisBase):
    requires = ['']
    sets = ['stim_dff_%s'%cs for cs in config.stimuli()] + \
           ['stim_dff_2_4_%s'%cs for cs in config.stimuli()] + \
           ['stim_dff_alltrials_%s'%cs for cs in config.stimuli()]
    across = 'day'
    updated = '180831'

    def run(self, date):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        date : Date object

        Returns
        -------
        dict
            All of the output values

        """

        out = self.nanoutput()

        for cs in config.stimuli():
            out['stim_dff_%s' % cs] = self.get_stimuli(date.runs('training'), cs, (0, 2), 'dff', False)
            out['stim_dff_2_4_%s'%cs] = self.get_stimuli(date.runs('training'), cs, (2, 4), 'dff', False)
            out['stim_dff_alltrials_%s' % cs] = self.get_stimuli(date.runs('training'), cs, (0, 2), 'dff', True)

        return out

    def get_stimuli(self, runs, cs, trange, ttype, all=False):
        """
        Get stimulus responses from training runs

        Parameters
        ----------
        runs : RunSorter
            Contains all Run objects
        cs : str
            Stimulus name
        trange : tuple of ints
            Time range to average over
        ttype : str
            Trace type, 'dff' or 'deconvolved'
        all : bool
            If true, include time after licking and miss trials

        Returns
        -------
        numpy array
            Vector of mean responses of length ncells
        """

        err = -1 if all else 0
        lick = -1 if all else 100
        baseline = (-1, 0) if 'dec' not in ttype else (-1, -1)

        # Go through the added stimuli and add all onsets
        trs = []
        for run in runs:
            with warnings.catch_warnings():
                t2p = run.trace2p()
                cstrs = t2p.cstraces(cs, trange[0], trange[1],
                                     ttype, lick, err, baseline)

                warnings.simplefilter('ignore', category=RuntimeWarning)

                if len(trs) == 0:
                    trs = np.nanmean(cstrs, axis=1)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, np.nanmean(cstrs, axis=1)], axis=1)

                if trange[1] <= 2 and cs == 'plus':
                    pav = t2p.cstraces('pavlovian', trange[0], trange[1],
                                       ttype, lick, err, baseline)
                    trs = np.concatenate([trs, np.nanmean(pav, axis=1)], axis=1)

        return np.nanmean(trs, axis=1)