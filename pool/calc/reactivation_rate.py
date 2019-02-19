"""Analyses directly related to what cells are driven by."""
import numpy as np
from scipy import stats
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from . import good


@memoize(across='date', updated=190130, returns='value')
def freq(date, cs, state='sated', classifier_threshold=0.1):
    """
    Return the frequency of reactivation averaged across times of inactivity.

    Parameters
    ----------
    date : Date object
    cs : str, stimulus
    state : str {'sated', 'hungry', 'all'}
    classifier_threshold : float

    Returns
    -------
    Frequency per day
    """

    # Only calculate if reactivation can be trusted
    if not good.reactivation(date, cs):
        return None

    reps = 0.0
    nframes = 0.0
    framerate = 0.0

    runs = (date.runs('spontaneous') if state == 'all'
            else date.runs('spontaneous', tags='state'))

    for run in runs:
        t2p = run.trace2p()
        framerate = t2p.framerate
        trs = t2p.traces('deconvolved')
        mask = t2p.inactivity()

        c2p = run.classify2p()
        evs = c2p.events(cs, classifier_threshold, trs, mask=mask, xmask=True)

        reps += len(evs)
        nframes += np.sum(mask)

    # Catch when we can't find the events
    if nframes == 0:
        return None

    return reps/nframes*framerate
