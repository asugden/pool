"""Analyses directly related to what cells are driven by."""
import numpy as np
from scipy import stats
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from .. import stimulus


@memoize(across='date', updated=190130, returns='cell array')
def visually(date, cs, integrate_bins=6, ncses=3):
    """
    Calculate the probability of being visually driven for each cell.

    Parameters
    ----------
    date : Date
    cs : str
    integrate_bins : int
        Number of bins over which to integrate the visual stim.
    ncses : int
        Number of possible cses, used to correct for multiple comparisons.
    nolick : bool
        If True, exclude trials where there was licking during the stimulus
        presentation.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    # Baseline is mean across frames, now ncells x nonsets
    baselines = nanmean(stimulus.trials(date, cs, start_s=-1, end_s=0), axis=1)
    stimuli = stimulus.trials(date, cs, start_s=0)
    fintegrate = -(-np.shape(stimuli)[1]//integrate_bins)  # ceiling division

    # Per-cell value
    meanbl = nanmean(baselines, axis=1)
    ncells = np.shape(baselines)[0]

    # We will save the maximum inverse p values
    maxinvps = np.zeros(ncells, dtype=np.float64)
    bonferroni_n = ncells*ncses*integrate_bins

    for i in range(integrate_bins):
        trs = nanmean(stimuli[:, i*fintegrate:(i+1)*fintegrate, :], axis=1)

        for c in range(ncells):
            if nanmean(trs[c, :]) > meanbl[c]:
                pv = stats.ranksums(baselines[c, :], trs[c, :]).pvalue
                logpv = -1*np.log(pv/bonferroni_n)
                if logpv > maxinvps[c]:
                    maxinvps[c] = logpv

    return maxinvps

@memoize(across='date', updated=190130, returns='cell array')
def visually_inhib(date, cs, integrate_bins=6, ncses=3):
    """
    Calculate the probability of being visually driven for each cell.

    Parameters
    ----------
    date : Date
    cs : str
    integrate_bins : int
        Number of bins over which to integrate the visual stim.
    ncses : int
        Number of possible cses, used to correct for multiple comparisons.
    nolick : bool
        If True, exclude trials where there was licking during the stimulus
        presentation.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    # Baseline is mean across frames, now ncells x nonsets
    baselines = nanmean(stimulus.trials(date, cs, start_s=-1, end_s=0)*-1, axis=1)
    stimuli = stimulus.trials(date, cs, start_s=0)*-1
    fintegrate = -(-np.shape(stimuli)[1]//integrate_bins)  # ceiling division

    # Per-cell value
    meanbl = nanmean(baselines, axis=1)
    ncells = np.shape(baselines)[0]

    # We will save the maximum inverse p values
    maxinvps = np.zeros(ncells, dtype=np.float64)
    bonferroni_n = ncells*ncses*integrate_bins

    for i in range(integrate_bins):
        trs = nanmean(stimuli[:, i*fintegrate:(i+1)*fintegrate, :], axis=1)

        for c in range(ncells):
            if nanmean(trs[c, :]) > meanbl[c]:
                pv = stats.ranksums(baselines[c, :], trs[c, :]).pvalue
                logpv = -1*np.log(pv/bonferroni_n)
                if logpv > maxinvps[c]:
                    maxinvps[c] = logpv

    return maxinvps


@memoize(across='date', updated=190218, returns='cell array')
def visually_classic(date, cs, integrate_secs=0.3, ncses=3, firstlick=True):
    """
    Calculate the probability of being visually driven for each cell.

    Parameters
    ----------
    date : Date
    cs : str
    integrate_secs : float
        Number of seconds over which to integrate the visual stim.
    ncses : int
        Number of possible cses, used to correct for multiple comparisons.
    nolick : bool
        If True, exclude trials where there was licking during the stimulus
        presentation.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    # Baseline is mean across frames, now ncells x nonsets
    baselines = nanmean(stimulus.trials(date, cs, start_s=-1, end_s=0), axis=1)

    if firstlick:
        stimuli = stimulus.trials(date, cs, start_s=0, end_s=None,
                                  error_trials=0, cutoff_before_lick_ms=100)
    else:
        stimuli = stimulus.trials(date, cs, start_s=0)

    # Get the median first lick
    runs = date.runs('training', tags='hungry')
    framerate = runs[0].trace2p().framerate
    fintegrate = int(round(integrate_secs*framerate))
    ts = np.arange(0, np.shape(stimuli)[1]+0.5, fintegrate)

    if firstlick:
        mfl = median_firstlick(runs, cs)

        # Cut off the first number after the median first lick
        am = np.argmax(ts > mfl)
        if np.max(ts) > mfl and am < len(ts) - 1:
            ts = ts[:am + 1]

    integrate_bins = len(ts) - 1

    # Per-cell value
    meanbl = nanmean(baselines, axis=1)
    ncells = np.shape(baselines)[0]

    # We will save the maximum inverse p values
    maxinvps = np.zeros(ncells, dtype=np.float64)

    for i in range(integrate_bins):
        trs = nanmean(stimuli[:, i*fintegrate:(i+1)*fintegrate, :], axis=1)

        for c in range(ncells):
            if nanmean(trs[c, :]) > meanbl[c]:
                pv = stats.ranksums(baselines[c, :], trs[c, :]).pvalue
                logpv = -1*np.log(pv)
                if logpv > maxinvps[c]:
                    maxinvps[c] = logpv

    return maxinvps


def median_firstlick(runs, cs):
    """
    Get the amount of total running and the running per CS
    :param data:
    :return:
    """

    firstlicks = []
    for run in runs:
        t2p = run.trace2p()
        fl = t2p.firstlick(cs, units='frames', maxframes=t2p.framerate*2)
        fl[np.isnan(fl)] = int(round(t2p.framerate*2))
        if len(firstlicks) == 0:
            firstlicks = fl
        else:
            firstlicks = np.concatenate([firstlicks, fl], axis=0)

    if len(firstlicks) < 2:
        return int(round(t2p.framerate*2))
    else:
        return np.nanmedian(firstlicks)


@memoize(across='date', updated=190118, returns='cell array')
def trial(date, cs, integrate_bins=10, ncses=3):
    """
    Calculate the probability of being trial-driven for each cell.

    Parameters
    ----------
    date : Date
    cs : str
    integrate_bins : int
        Number of bins over which to integrate the visual stim.
    ncses : int
        Number of possible cses, used to correct for multiple comparisons.
    nolick : bool
        If True, exclude trials where there was licking during the stimulus
        presentation.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are the log
        inverse p-value of that cell responding to the particular cs.

    """

    # Baseline is mean across frames, now ncells x nonsets
    baselines = nanmean(stimulus.trials(date, cs, start_s=-1, end_s=0), axis=1)
    stimuli = stimulus.trials(date, cs, start_s=0, end_relative=2)
    fintegrate = -(-np.shape(stimuli)[1]//integrate_bins)  # ceiling division

    # Per-cell value
    meanbl = nanmean(baselines, axis=1)
    ncells = np.shape(baselines)[0]

    # We will save the maximum inverse p values
    maxinvps = np.zeros(ncells, dtype=np.float64)
    bonferroni_n = ncells*ncses*integrate_bins

    for i in range(integrate_bins):
        trs = nanmean(stimuli[:, i*fintegrate:(i+1)*fintegrate, :], axis=1)

        for c in range(ncells):
            if nanmean(trs[c, :]) > meanbl[c]:
                pv = stats.ranksums(baselines[c, :], trs[c, :]).pvalue
                logpv = -1*np.log(pv/bonferroni_n)
                if logpv > maxinvps[c]:
                    maxinvps[c] = logpv

    return maxinvps
