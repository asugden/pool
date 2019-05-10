"""Analyses directly related to classifier randomization."""
from __future__ import division

from ..database import memoize


@memoize(across='date', updated=190509)
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
    runs = date.runs(run_types=run_types, tags=tags)
    for run in runs:
        c2p = run.classify2p(newpars=added_pars)
        rand = c2p.randomization(
            randomization_type, mask_running=mask_running,
            mask_licking=mask_licking, mask_motion=mask_motion)

        nrunreal, nrunrand = rand.real_false_positives(cs, threshold=threshold)
        nreal += nrunreal
        nrand += nrunrand

    return nreal, nrand
