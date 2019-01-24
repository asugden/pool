"""Analyses directly related to stimulus presentation."""
from builtins import range
import numpy as np
from scipy import stats

from .imaging import framerate
from ..database import memoize


@memoize(across='date', updated=190118)
def vdrive(date, cs, integrate_s=0.3, pval=0.05, ncses=3, nolick=True):
    """
    Calculate the probability of being visually driven for each cell.

    Parameters
    ----------
    date : Date
    cs : str
    integrate_s : float
        Time in seconds to integrate over at the onset of the visual stim.
    pval : float
        Not used?
    ncses : int
        Number of possible cses, used to correct for multiple comparisons.
    nolick : bool
        If True, exclude trials where there was licking during the stimulus
        presentation.

    Result
    ------
    np.ndarray
        An attay of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """
    runs = date.runs('training')
    fr = framerate(date)
    integrate_fr = int(round(integrate_s*fr))

    # Calculate the median first lick frame.
    firstlicks = []
    for run in runs:
        t2p = run.trace2p()
        fl = t2p.firstlick(cs, units='frames', maxframes=fr*2)
        fl[np.isnan(fl)] = int(round(fr*2))
        firstlicks = np.concatenate([firstlicks, fl], axis=0)

    if len(firstlicks) < 2:
        mfl = int(round(t2p.framerate*2))
    else:
        mfl = np.nanmedian(firstlicks)

    # Cut off the first number after the median first lick
    ts = np.arange(0, 2*fr+1, integrate_fr)
    am = np.argmax(ts > mfl)
    if np.max(ts) > mfl and am < len(ts) - 1:
        ts = ts[:am+1]

    bls = _gettrials(runs, cs, start=-1, end=0, error_trials=-1, lick=-1)
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
        trs = _gettrials(runs, cs, start=start, end=end,
                         error_trials=0, lick=100 if not nolick else -1)

        for c in range(np.shape(trs)[0]):
            if np.nanmean(trs[c, :]) > meanbl[c]:
                pv = stats.ranksums(bls[c, :], trs[c, :]).pvalue
                logpv = -1*np.log(stats.ranksums(bls[c, :], trs[c, :]).pvalue)
                if logpv > maxinvps[c]:
                    maxinvps[c] = logpv
                if pv <= pval:
                    vdriven[c] = True

    return maxinvps


def _gettrials(runs, cs, start=0, end=0, error_trials=-1, lick=-1):
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
            trs = t2p.cstraces(
                cs, start_s=start, end_s=end, trace_type='dff',
                cutoff_before_lick_ms=lick, errortrials=error_trials)
            if cs == 'plus':
                pavs = t2p.cstraces(
                    'pavlovian', start_s=start, end_s=end, trace_type='dff',
                    cutoff_before_lick_ms=lick, errortrials=error_trials)
                trs = np.concatenate([trs, pavs], axis=2)

            if len(alltrs) == 0:
                alltrs = trs
            else:
                alltrs = np.concatenate([alltrs, trs], axis=2)

        alltrs = np.nanmean(alltrs, axis=1)  # across frames
        return alltrs
