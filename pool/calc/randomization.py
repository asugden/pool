"""Analyses directly related to classifier randomization."""
from __future__ import division
import numpy as np

from flow.classify2p import NoClassifierError
from ..database import memoize


@memoize(across='date', updated=190516)
def false_positive_rate(
        date, cs, randomization_type='identity', classifier='aode',
        match_activity=False, threshold=0.1, run_types='spontaneous',
        tags='sated', mask_running=True, mask_licking=True, mask_motion=True):
    """
    Return the per-cs false-positive rate for a particular randomization type.

    Parameters
    ----------
    date : Date
    cs : str
        Stimulus type {'plus', 'neutral', 'minus'}
    randomization_type : str {'time', 'identity'}
        Randomize in time or identity
    classifier : {'aode', 'naive-bayes'}
        Classifier to use.
    match_activity : bool
        If True, match the activity of all cells across time.

    Returns
    -------
    dict of floats
        False positive rate per cs

    """

    added_pars = {'classifier': classifier}
    if match_activity:
        added_pars['equalize-cell-activity'] = True

    nreal, nrand = 0, 0
    non_nan_events = False
    runs = date.runs(run_types=run_types, tags=tags)
    for run in runs:
        try:
            c2p = run.classify2p(newpars=added_pars)
            rand = c2p.randomization(
                randomization_type, mask_running=mask_running,
                mask_licking=mask_licking, mask_motion=mask_motion)
        except NoClassifierError:
            continue

        nrunreal, nrunrand = rand.real_false_positives(cs, threshold=threshold)
        if ~np.isnan(nrunreal):
            nreal += nrunreal
            non_nan_events = True
        if ~np.isnan(nrunrand):
            nrand += nrunrand
            non_nan_events = True

    # If the only non-NaN results across runs is (0, 0), that is fine and
    # return (0, 0), but if all the results are NaN, return (NaN, NaN)
    if non_nan_events:
        return nreal, nrand
    else:
        return np.nan, np.nan


@memoize(across='date', updated=190423)
def false_positive_distribution(
        date, cs, randomization_type='identity', classifier='aode',
        match_activity=False):
    """
    Return the per-cs false-positive rate for a particular randomization type.

    Parameters
    ----------
    date : Date
    cs : str
        Stimulus type {'plus', 'neutral', 'minus'}
    randomization_type : str {'time', 'identity'}
        Randomize in time or identity
    classifier : {'aode', 'naive-bayes'}
        Classifier to use.
    match_activity : bool
        If True, match the activity of all cells across time.

    Returns
    -------
    dict of floats
        False positive rate per cs

    """

    added_pars = {'classifier': classifier}
    if match_activity:
        added_pars['equalize-cell-activity'] = True

    bin_mins = np.arange(0.05, 1.0, 0.05)
    nreal, nrand = np.zeros(len(bin_mins)), np.zeros(len(bin_mins))
    runs = date.runs(run_types='spontaneous', tags='sated')
    for run in runs:
        c2p = run.classify2p(newpars=added_pars)
        rand = c2p.randomization(randomization_type)

        for i, bmin in enumerate(bin_mins):
            nrunreal, nrunrand = rand.real_false_positives(cs, threshold=bmin)
            nreal[i] += nrunreal
            nrand[i] += nrunrand

    out = np.array([bin_mins, nreal, nrand])
    return out
