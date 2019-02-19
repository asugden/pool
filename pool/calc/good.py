"""Analyses determining if a day or recording is good."""
import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from .. import stimulus

@memoize(across='date', updated=181003, returns='value')
def reactivation(date, cs):
    """

    Parameters
    ----------
    date
    cs

    Returns
    -------

    """

    # ncells, ntimes, nonsets
    trs = stimulus.trials(date, cs, start_s=0, end_s=None,
                          trace_type='dff', baseline=(-1, 0))

    trs = nanmean(trs, axis=2)
    trs = nanmean(trs, axis=1)

    # Across ncells
    dff_active = np.sum(trs > 0.025)/float(len(trs))

    return dff_active > 0.05
