"""Measures of population distance."""
import numpy as np
from scipy.spatial.distance import cosine
try:
    from bottleneck import nanmean, nanmedian
except ImportError:
    from numpy import nanmean, nanmedian

from ..database import memoize


@memoize(across='date', updated=190220, returns='cell array')
def outliers(date, trace_type='deconvolved', sigma=2, run_type='spontaneous'):
    """
    Find the cells with outlier activity levels.

    Parameters
    ----------
    date : Date instance
    trace_type : str {'dff', 'deconvolved', 'raw'}
    sigma : int
        The number of standard deviations beyond the median to include
    run_type : str
        The type of runs to analyze, traditionally spontaneous

    Returns
    -------
    array of bool
        True means that the cell is an outlier
    """

    outliers = None
    for run in date.runs(run_type):
        t2p = run.trace2p()
        fmin = t2p.lastonset()
        trs = t2p.trace(trace_type)[:, fmin:]

        cellact = nanmean(trs, axis=1)
        outs = cellact > nanmedian(cellact) + sigma*np.std(cellact)

        if outliers is None:
            outliers = outs
        else:
            outliers = np.bitwise_or(outliers, outs)

    return outliers


def keep(date, trace_type='deconvolved', sigma=2, run_type='spontaneous'):
    """
    Determine the cells that should be kept for analyses.

    Parameters
    ----------
    date : Date instance
    trace_type : str {'dff', 'deconvolved', 'raw'}
    sigma : int
        The number of standard deviations beyond the median to include
    run_type : str
        The type of runs to analyze, traditionally spontaneous

    Returns
    -------
    array of bool
        True means that the cell should be kept
    """

    return np.invert(outliers(date, trace_type, sigma, run_type))
