"""Analyses directly related to what cells are driven by."""
import numpy as np
from scipy import stats
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from . import good
from . import clusters


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
def count_cell(date, cs, state='sated', classifier_threshold=0.1, deconvolved_threshold=0.2):
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
def count_pair(date, cs, state='sated', classifier_threshold=0.1, deconvolved_threshold=0.2):
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


@memoize(across='run', updated=190605, returns='values')
def events(run, cs, classifier_threshold=0.1):
    """
    Return the frequency of reactivation averaged across times of inactivity.

    Parameters
    ----------
    run : Run object
    cs : str, stimulus
    classifier_threshold : float

    Returns
    -------
    Frequency per day
    """

    # Only calculate if reactivation can be trusted
    if not good.reactivation(run.parent, cs):
        return None

    t2p = run.trace2p()
    trs = t2p.trace('deconvolved')
    mask = t2p.inactivity()

    c2p = run.classify2p()
    evs = c2p.events(cs, classifier_threshold, trs, mask=mask, xmask=True)

    return evs


@memoize(across='run', updated=190605, returns='values')
def event_rich_poor(run,
                    cs='plus',
                    classifier_threshold=0.1,
                    deconvolved_threshold=0.2,
                    trange=(-2, 3),
                    visual_drivenness=50,
                    correlation='noise',
                    reward_cells_required=1):
    """
    Return the rich-poor value for each event.

    Parameters
    ----------
    run
    cs
    classifier_threshold
    deconvolved_threshold
    trange
    visual_drivenness
    correlation
    reward_cells_required

    Returns
    -------

    """

    rclbls = clusters.reward(run.parent,
                             visual_drivenness=visual_drivenness,
                             correlation=correlation,
                             reward_cells_required=reward_cells_required)
    nclbls = clusters.nonreward(run.parent,
                                visual_drivenness=visual_drivenness,
                                correlation=correlation,
                                reward_cells_required=reward_cells_required)

    evs = events(run, cs, classifier_threshold=classifier_threshold)
    if evs is None:
        return None

    t2p = run.trace2p()
    trs = t2p.trace('deconvolved')
    reward_non = []

    ncells = np.shape(trs)[0]

    for ev in evs:
        if -1*trange[0] < ev < np.shape(trs)[1] - trange[1]:
            act = np.nanmax(trs[:, ev+trange[0]:ev+trange[1]], axis=1)
            act = act > deconvolved_threshold

            ract = np.sum(np.bitwise_and(act, rclbls)).astype(np.float64)
            nact = np.sum(np.bitwise_and(act, nclbls)).astype(np.float64)
            rn = np.sum(ract)/(np.sum(ract) + np.sum(nact))
            reward_non.append(rn)

    return reward_non
