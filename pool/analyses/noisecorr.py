import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

from . import base
from pool import config


class Noisecorr(base.AnalysisBase):
    requires = ['']
    sets = ['noisecorr_%s'%cs for cs in config.stimuli()] + \
           ['noisecorr_hofer']
    across = 'day'
    updated = '181031'

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

        all = None
        for cs in config.stimuli():
            trs = self.get_stimuli(date.runs('training'), cs, (0, 2), 'dff')
            all = trs if all is None else np.concatenate([all, (trs.T - np.nanmean(trs, axis=1)).T], axis=1)
            out['noisecorr_%s'%cs] = self.calc_noisecorr_cohen(trs)

        out['noisecorr_hofer'] = self.calc_noisecorr_hofer(all)

        return out

    @staticmethod
    def get_stimuli(runs, cs, trange=(0, 2), ttype='dff', nolick=False):
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
        nolick : bool
            If true, remove time after licking

        Returns
        -------
        numpy array
            Vector of mean responses of length ncells
        """

        err = -1
        lick = -1 if all else 100
        baseline = (-1, 0) if ttype == 'dff' else (-1, -1)

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

        return trs

    @staticmethod
    def calc_noisecorr_cohen(trs, includes_nans=False, nrand=500):
        """
        Calculate the noise correlations of cells via their traces.
        Uses the method of Marlene Cohen and John Maunsell.

        Parameters
        ----------
        trs : matrix of ncells x ntrials
            Traces of ncells x ntrials
        includes_nans : bool
            If true, use Pandas to calculate the correlation to account for NaNs
        nrand : int
            Number of randomizations to subtract

        Returns
        -------
        matrix of ncells x ncells
            Noise correlations
        """

        if np.shape(trs)[1] < 10:
            return np.nan
        else:
            if not includes_nans:
                out = np.corrcoef(trs)
            else:
                dftrs = pd.DataFrame(trs.T)
                out = dftrs.corr().as_matrix()

            ncells = np.shape(trs)[0]
            stimorder = np.arange(np.shape(trs)[1])
            for i in range(nrand):
                for c in range(ncells):
                    np.random.shuffle(stimorder)
                    trs[c, :] = trs[c, stimorder]

                if not includes_nans:
                    out -= np.corrcoef(trs)/float(nrand)
                else:
                    dftrs = pd.DataFrame(trs.T)
                    out -= dftrs.corr().as_matrix()/float(nrand)

        return out

    @staticmethod
    def calc_noisecorr_hofer(trs, includes_nans=False):
        """
        Calculate the noise correlations of cells via their traces.
        Requires that the mean stimulus response has been subtracted.
        Uses the method of Sonja Hofer.

        Parameters
        ----------
        trs : matrix of ncells x ntrials
            Traces of ncells x ntrials
        includes_nans : bool
            If true, use Pandas to calculate the correlation to account for NaNs

        Returns
        -------
        matrix of ncells x ncells
            Noise correlations
        """

        if not includes_nans:
            return np.corrcoef(trs)
        else:
            dftrs = pd.DataFrame(trs.T)
            return dftrs.corr().as_matrix()
