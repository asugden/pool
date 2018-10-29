import numpy as np
from scipy import stats

from . import base
from pool import config


class Vdrive(base.AnalysisBase):
    requires = ['']
    sets = ['vdrive_%s'%cs for cs in config.stimuli()] + \
           ['vdrive_fraction_%s'%cs for cs in config.stimuli()]
    across = 'day'
    updated = '180911'

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
            out['vdrive_%s' % cs] = self.responses(date.runs('training'), cs)
            out['vdrive_fraction_%s' % cs] = np.nansum(out['vdrive_%s'%cs] > 50)/float(len(out['vdrive_%s'%cs]))

        return out

    def median_first_lick(self, runs, cs):
        """
        Get the median first lick

        Parameters
        ----------
        runs : RunSorter
        cs : str
            Stimulus name

        Returns
        -------
        float
            Median first lick time in frames
        """

        firstlicks = []
        for run in runs:
            t2p = run.trace2p()
            fl = t2p.firstlick(cs, units='frames', maxframes=t2p.framerate*2)
            fl[np.isnan(fl)] = int(round(t2p.framerate*2))
            firstlicks = np.concatenate([firstlicks, fl], axis=0)

        if len(firstlicks) < 2: return int(round(t2p.framerate*2))
        return np.nanmedian(firstlicks)

    @staticmethod
    def gettrials(runs, cs, start=0, end=0, error_trials=-1, lick=-1):
        """
        Get all training trials.

        Parameters
        ----------
        runs : RunSorter
        cs : str
            Stimulus name
        start : float
            Beginning of time to integrate
        end : float
            End of time to integrate
        error_trials : int
            -1 all trials, 0 correct trials, 1 error trials
        lick : float
            Number of milliseconds to cut off before the first lick

        Returns
        -------
        numpy matrix
            All trials of size ncells, ntrials

        """

        alltrs = []
        for run in runs:
            t2p = run.trace2p()

            # ncells, frames, nstimuli/onsets
            trs = t2p.cstraces(cs, start_s=start, end_s=end, trace_type='dff',
                               cutoff_before_lick_ms=lick, errortrials=error_trials)
            if cs == 'plus':
                pavs = t2p.cstraces('pavlovian', start_s=start, end_s=end, trace_type='dff',
                                    cutoff_before_lick_ms=lick, errortrials=error_trials)
                trs = np.concatenate([trs, pavs], axis=2)

            if len(alltrs) == 0:
                alltrs = trs
            else:
                alltrs = np.concatenate([alltrs, trs], axis=2)

        alltrs = np.nanmean(alltrs, axis=1)  # across frames
        return alltrs

    def responses(self, runs, cs, tintegrate=0.3, pval=0.05, ncses=3, nolick=True):

        mfl = self.median_first_lick(runs, cs)
        fr = self.analysis('framerate')
        fintegrate = int(round(tintegrate*fr))

        # Cut off the first number after the median first lick
        ts = np.arange(0, 2*fr+1, fintegrate)
        am = np.argmax(ts > mfl)
        if np.max(ts) > mfl and am < len(ts) - 1: ts = ts[:am+1]

        bls = self.gettrials(runs, cs, start=-1, end=0, error_trials=-1, lick=-1)
        meanbl = np.nanmean(bls, axis=1)

        vdriven = np.zeros(np.shape(bls)[0], dtype=bool)
        pval /= len(ts) - 1  # Correct for number of time points
        pval /= np.shape(bls)[0]  # Correct for the number of cells
        pval /= ncses  # Correct for number of CSes

        # We will save the maximum inverse p values
        maxinvps = np.zeros(np.shape(bls)[0], dtype=np.float64)

        for i in range(len(ts) - 1):
            start = float(ts[i])/fr
            end = float(ts[i+1])/fr
            trs = self.gettrials(runs, cs, start=start, end=end,
                                 error_trials=0, lick=100 if not nolick else -1)

            for c in range(np.shape(trs)[0]):
                if np.nanmean(trs[c, :]) > meanbl[c]:
                    pv = stats.ranksums(bls[c, :], trs[c, :]).pvalue
                    logpv = -1*np.log(stats.ranksums(bls[c, :], trs[c, :]).pvalue)
                    if logpv > maxinvps[c]: maxinvps[c] = logpv
                    if pv <= pval:
                        vdriven[c] = True

        return maxinvps
