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
