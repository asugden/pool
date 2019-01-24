import numpy as np
import pandas as pd
import warnings

from . import base
from pool import config


class Stim(base.AnalysisBase):
    requires = ['']
    sets = ['stim_dff_%s' % cs for cs in config.stimuli()] + \
           ['stim_dff_2_4_%s' % cs for cs in config.stimuli()] + \
           ['stim_dff_alltrials_%s' % cs for cs in config.stimuli()] + \
           ['stim_dff_alltrials_pertrial']
    across = 'day'
    updated = '1901152'

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
            out['stim_dff_%s' % cs] = self.get_stimuli(
                date.runs('training'), cs, (0, 2), 'dff', all_trials=False)
            out['stim_dff_2_4_%s' % cs] = self.get_stimuli(
                date.runs('training'), cs, (2, 4), 'dff', all_trials=False)
            out['stim_dff_alltrials_%s' % cs] = self.get_stimuli(
                date.runs('training'), cs, (0, 2), 'dff', all_trials=True)
        out['stim_dff_alltrials_pertrial'] = self.get_stimuli_per_trial(
            date.runs('training'), (0, 2), 'dff')

        return out

    def get_stimuli(self, runs, cs, trange, ttype, all_trials=False):
        """
        Get stimulus responses from training runs.

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
        all_trials : bool
            If true, include time after licking and miss trials

        Returns
        -------
        numpy array
            Vector of mean responses of length ncells.

        """

        err = -1 if all_trials else 0
        lick = -1 if all_trials else 100
        baseline = (-1, 0) if 'dec' not in ttype else (-1, -1)

        # Go through the added stimuli and add all onsets
        trs = []
        for run in runs:
            with warnings.catch_warnings():
                t2p = run.trace2p()
                # cstrs = ncells, frames, nstimuli/onsets
                cstrs = t2p.cstraces(cs, trange[0], trange[1],
                                     ttype, lick, err, baseline)

                warnings.simplefilter('ignore', category=RuntimeWarning)

                # if len(trs) == 0:
                #     trs = np.nanmean(cstrs, axis=1)  # ncells, frames, nstimuli/onsets
                # else:
                #     trs = np.concatenate([trs, np.nanmean(cstrs, axis=1)], axis=1)
                trs.append(np.nanmean(cstrs, axis=1))

                # Include pavlovian trials with plus trials
                if trange[1] <= 2 and cs == 'plus':
                    pav = t2p.cstraces('pavlovian', trange[0], trange[1],
                                       ttype, lick, err, baseline)
                    # trs = np.concatenate([trs, np.nanmean(pav, axis=1)], axis=1)
                    trs.append(np.nanmean(pav, axis=1))

        trs_cat = np.concatenate(trs, axis=1)

        return np.nanmean(trs_cat, axis=1)

    def get_stimuli_per_trial(self, runs, trange, ttype):

        lick = -1
        err = -1
        baseline = (-1, 0) if 'dec' not in ttype else (-1, -1)

        result = [pd.DataFrame()]
        for run in runs:
            t2p = run.trace2p()
            # cstrs = ncells, frames, nstimuli/onsets
            cstrs = t2p.cstraces(
                '', trange[0], trange[1], ttype, lick, err, baseline)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                trs = np.nanmean(cstrs, axis=1)
            for roi_idx, roi_response in enumerate(trs):
                index = pd.MultiIndex.from_product([
                    [run.mouse], [run.date], [run.run], [roi_idx],
                    range(len(roi_response))],
                    names=['mouse', 'date', 'run', 'roi_idx', 'trial_idx'])
                result.append(
                    pd.DataFrame({'response': roi_response}, index=index))

        df = pd.concat(result, axis=0)

        return df
