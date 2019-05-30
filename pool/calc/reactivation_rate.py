"""Analyses directly related to what cells are driven by."""
import numpy as np
from scipy import stats
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from . import good


@memoize(across='date', updated=190402, returns='value')
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

    runs = date.runs('spontaneous') if state == 'all' \
        else date.runs('spontaneous', tags=[state])

    for run in runs:
        t2p = run.trace2p()
        framerate = t2p.framerate
        trs = t2p.trace('deconvolved')
        mask = t2p.inactivity()

        c2p = run.classify2p()
        evs = c2p.events(cs, classifier_threshold, trs, mask=mask, xmask=True)

        reps += len(evs)
        nframes += np.sum(mask)

    # Catch when we can't find the events
    if nframes == 0:
        return None

    return reps/nframes*framerate

@memoize(across='date', updated=190530, returns='value')
def freq_denominator(date, state='sated', classifier_threshold=0.1):
    """
    Return the frequency of reactivation averaged across times of inactivity.

    Parameters
    ----------
    date : Date object
    state : str {'sated', 'hungry', 'all'}
    classifier_threshold : float

    Returns
    -------
    Frequency per day
    """

    # Only calculate if reactivation can be trusted
    nframes = 0.0
    framerate = 0.0

    runs = date.runs('spontaneous') if state == 'all' \
        else date.runs('spontaneous', tags=[state])

    for run in runs:
        t2p = run.trace2p()
        framerate = t2p.framerate
        mask = t2p.inactivity()
        nframes += np.sum(mask > 0)

    # Catch when we can't find the events
    if nframes == 0:
        return None

    return nframes/framerate

@memoize(across='date', updated=190423, returns='cell array')
def count(date, cs, state='sated', classifier_threshold=0.1, deconvolved_threshold=0.2):
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

    nframes = 0.0

    runs = date.runs('spontaneous') if state == 'all' \
        else date.runs('spontaneous', tags=[state])

    rep = None

    for run in runs:
        t2p = run.trace2p()

        if rep is None:
            rep = np.zeros(t2p.ncells)

        trs = t2p.trace('deconvolved')
        mask = t2p.inactivity()

        c2p = run.classify2p()
        frange = c2p.frame_range
        evs = c2p.events(cs, classifier_threshold, trs, mask=mask, xmask=True)

        for ev in evs:
            act = np.nanmax(trs[:, ev + frange[0]:ev + frange[1]], axis=1)
            act = act > deconvolved_threshold

            rep += act.astype(np.float64)

        nframes += np.sum(mask)

    # Catch when we can't find the events
    if nframes == 0:
        return None

    return rep


@memoize(across='date', updated=190423, returns='cell matrix')
def pair(date, cs, state='sated', classifier_threshold=0.1, deconvolved_threshold=0.2):
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

    nframes = 0.0

    runs = date.runs('spontaneous') if state == 'all' \
        else date.runs('spontaneous', tags=[state])

    pairrep = None

    for run in runs:
        t2p = run.trace2p()

        if pairrep is None:
            pairrep = np.zeros((t2p.ncells, t2p.ncells))

        trs = t2p.trace('deconvolved')
        mask = t2p.inactivity()

        c2p = run.classify2p()
        frange = c2p.frame_range
        evs = c2p.events(cs, classifier_threshold, trs, mask=mask, xmask=True)

        for ev in evs:
            act = np.nanmax(trs[:, ev + frange[0]:ev + frange[1]], axis=1)
            act = act > deconvolved_threshold

            actout = np.zeros((len(act), len(act)), dtype=np.bool)  # pairs of cells that overlapped
            for i in range(len(act)):  # iterate over cells
                actout[i, :] = np.bitwise_and(act[i], act)

            pairrep += actout.astype(np.float64)

        nframes += np.sum(mask)

    # Catch when we can't find the events
    if nframes == 0:
        return None

    return pairrep
