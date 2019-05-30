"""Analyses directly related to classifier randomization."""
from __future__ import division
import numpy as np

from ..database import memoize


@memoize(across='date', updated=190002)
def false_positive_rate(
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

    nreal, nrand = 0, 0
    runs = date.runs(run_types='spontaneous', tags='sated')
    for run in runs:
        c2p = run.classify2p(newpars=added_pars)
        rand = c2p.randomization(randomization_type)

        nrunreal, nrunrand = rand.real_false_positives(cs, threshold=0.1)
        nreal += nrunreal
        nrand += nrunrand

    return nreal, nrand


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
