import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..database import memoize
from .. import stimulus

@memoize(across='date', updated=190207, returns='cell matrix')
def noise(date, cs, trange=(0, None), trace_type='dff',
          cutoff_before_lick_ms=-1, error_trials=-1,
          randomizations=500):
    """
    Create a matrix of the pairwise noise correlations between cells.

    Parameters
    ----------
    date : Date instance or RunSorter
    cs : str, stimulus
    trace_type : str {'dff', 'deconvolved'}
    trange : tuple of ints, time range to average
    exclude_licking : bool

    Returns
    -------
    matrix of floats, ncells x ncells
    """

    # ncells x frames x nstimuli/onsets
    trs = stimulus.trials(date, cs, start_s=trange[0], end_s=trange[1],
                              trace_type=trace_type,
                              cutoff_before_lick_ms=cutoff_before_lick_ms,
                              error_trials=error_trials)

    trs = np.nanmean(trs, axis=1)
    ncells = np.shape(trs)[0]
    corrs = np.zeros((ncells, ncells))
    corrs[:, :] = np.nan

    # Catch cases when there aren't enough trials
    if np.shape(trs)[1] < 10:
        return corrs

    stimorder = np.arange(np.shape(trs)[1])
    if np.sum(np.invert(np.isfinite(corrs))) == 0:
        corrs = np.corrcoef(trs)

        for i in range(randomizations):
            for c in range(ncells):
                np.random.shuffle(stimorder)
                trs[c, :] = trs[c, stimorder]

                corrs -= np.corrcoef(trs)/float(randomizations)
    else:
        dftrs = pd.DataFrame(trs.T)
        corrs = dftrs.corr().as_matrix()

        for i in range(randomizations):
            for c in range(ncells):
                np.random.shuffle(stimorder)
                trs[c, :] = trs[c, stimorder]

                dftrs = pd.DataFrame(trs.T)
                corrs -= dftrs.corr().as_matrix()/float(randomizations)

    return corrs
