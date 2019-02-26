"""Analyses directly related to behavioral performance."""
from __future__ import division
import numpy as np
from scipy.stats import norm

from ..database import memoize
from .. import engagement_hmm


@memoize(across='run', updated=190220)
def engaged(run, across_run=True):
    """
    Return result of engagement HMM.

    Parameters
    ----------
    run : Run
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    array of bool
        Boolean trial mask of engaged trials. True == engaged.

    """

    hmm = engagement_hmm.EngagementHMM()

    if across_run:
        date = run.parent
        runs = []
        for training_run in date.runs('training', tags=['hungry']):
            runs.append(training_run)

        return hmm.engagement(run)
    else:
        hmm.set_runs([run]).calculate()

        return hmm.engagement()


@memoize(across='date', updated=190226, returns='value')
def engagement(date):
    """
    Return result of engagement HMM.

    Parameters
    ----------
    date : Date

    Returns
    -------
    float
        Return the fraction of time engaged

    """

    hmm = engagement_hmm.EngagementHMM()
    hmm.set_runs(date.runs('training', tags='hungry')).calculate()
    return np.nanmean(hmm.engagement().astype(float))


@memoize(across='run', updated=190220)
def correct_count(run, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Number of correct trials of a specific cs type.

    Parameters
    ----------
    run : Run
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    int

    """
    errs = _trial_errors(
        run, cs=cs, hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    return int(np.sum(errs == 0))


@memoize(across='run', updated=190220)
def incorrect_count(run, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Number of incorrect trials of a specific cs type.

    Parameters
    ----------
    run : Run
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    int

    """
    errs = _trial_errors(
        run, cs=cs, hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    return int(np.sum(errs == 1))


@memoize(across='run', updated=190220)
def trial_count(run, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Number of trials of a specific cs type.

    Parameters
    ----------
    run : Run
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    int

    """
    errs = _trial_errors(
        run, cs=cs, hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    return len(errs)


@memoize(across='date', updated=190220)
def dprime(
        date, hmm_engaged=True, combine_pavlovian=False,
        combine_passives=True, across_run=True):
    """
    Return d-prime calculated for a specific date.

    Parameters
    ----------
    date : Date
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    combine_passives : bool
        If True, combine minus and neutral trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    nhits, nplus, nfas, npassives = 0, 0, 0, 0
    for run in date.runs(run_types=['training']):
        nhits += correct_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        nplus += trial_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        nfas += incorrect_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        npassives += trial_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        if combine_passives:
            nfas += incorrect_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)
            npassives += trial_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)

    z_hit_rate = norm.ppf((nhits + 0.5)/(nplus + 1.0))
    z_fa_rate = norm.ppf((nfas + 0.5)/(npassives + 1.0))

    return z_hit_rate - z_fa_rate


@memoize(across='run', updated=190220)
def dprime_run(
        run, hmm_engaged=True, combine_pavlovian=False,
        combine_passives=True, across_run=True):
    """
    Return d-prime calculated for a specific run.

    Parameters
    ----------
    run : Run
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    combine_passives : bool
        If True, combine minus and neutral trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    nhits, nplus, nfas, npassives = 0, 0, 0, 0
    nhits += correct_count(
        run, cs='plus', hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    nplus += trial_count(
        run, cs='plus', hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)

    nfas += incorrect_count(
        run, cs='minus', hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    npassives += trial_count(
        run, cs='minus', hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)

    if combine_passives:
        nfas += incorrect_count(
            run, cs='neutral', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        npassives += trial_count(
            run, cs='neutral', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

    z_hit_rate = norm.ppf((nhits + 0.5)/(nplus + 1.0))
    z_fa_rate = norm.ppf((nfas + 0.5)/(npassives + 1.0))

    return z_hit_rate - z_fa_rate


@memoize(across='date', updated=190131)
def criterion(
        date, hmm_engaged=True, combine_pavlovian=False,
        combine_passives=True, across_run=True):
    """
    Return criterion calculated for a specific date.

    Parameters
    ----------
    date : Date
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    combine_passives : bool
        If True, combine minus and neutral trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    nhits, nplus, nfas, npassives = 0, 0, 0, 0
    for run in date.runs(run_types=['training']):
        nhits += correct_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        nplus += trial_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        nfas += incorrect_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        npassives += trial_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        if combine_passives:
            nfas += incorrect_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)
            npassives += trial_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)

    z_hit_rate = norm.ppf((nhits + 0.5)/(nplus + 1.0))
    z_fa_rate = norm.ppf((nfas + 0.5)/(npassives + 1.0))

    return -1*(z_hit_rate + z_fa_rate)/2.


@memoize(across='date', updated=190131)
def likelihood_ratio(
        date, hmm_engaged=True, combine_pavlovian=False,
        combine_passives=True, across_run=True):
    """
    Return sdt likelihood-ratio calculated for a specific date.

    Parameters
    ----------
    date : Date
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    combine_passives : bool
        If True, combine minus and neutral trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    nhits, nplus, nfas, npassives = 0, 0, 0, 0
    for run in date.runs(run_types=['training']):
        nhits += correct_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        nplus += trial_count(
            run, cs='plus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        nfas += incorrect_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        npassives += trial_count(
            run, cs='minus', hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

        if combine_passives:
            nfas += incorrect_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)
            npassives += trial_count(
                run, cs='neutral', hmm_engaged=hmm_engaged,
                combine_pavlovian=combine_pavlovian, across_run=across_run)

    z_hit_rate = norm.ppf((nhits + 0.5)/(nplus + 1.0))
    z_fa_rate = norm.ppf((nfas + 0.5)/(npassives + 1.0))

    return np.exp((z_fa_rate**2 - z_hit_rate**2)/2.)


@memoize(across='date', updated=190130)
def dprime_percentile(
        date, hmm_engaged=True, combine_pavlovian=False,
        combine_passives=True, across_run=True):
    """
    Return d-prime percentile for a specific date.

    Parameters
    ----------
    date : Date
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    combine_passives : bool
        If True, combine minus and neutral trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    return norm.cdf(dprime(
        date, hmm_engaged=hmm_engaged, combine_pavlovian=combine_pavlovian,
        combine_passives=combine_passives, across_run=across_run))


@memoize(across='date', updated=190124)
def correct_fraction(date, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Fraction of correct trials of a specific cs type.

    Parameters
    ----------
    date : Date
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    ncorrect, ntrials = 0, 0
    for run in date.runs(run_types=['training']):
        ncorrect += correct_count(
            run, cs=cs, hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)
        ntrials += trial_count(
            run, cs=cs, hmm_engaged=hmm_engaged,
            combine_pavlovian=combine_pavlovian, across_run=across_run)

    if ntrials == 0:
        return np.nan

    return ncorrect/ntrials


@memoize(across='run', updated=190124)
def correct_fraction_run(
        run, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Fraction of correct trials of a specific cs type.

    Parameters
    ----------
    run : Run
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs.

    Returns
    -------
    float

    """
    ncorrect = correct_count(
        run, cs=cs, hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    ntrials = trial_count(
        run, cs=cs, hmm_engaged=hmm_engaged,
        combine_pavlovian=combine_pavlovian, across_run=across_run)
    return ncorrect/ntrials


def _trial_errors(run, cs=None, hmm_engaged=True, combine_pavlovian=False, across_run=True):
    """
    Return a boolean mask of incorrect trials of a specific type.

    Parameters
    ----------
    run : Run
    cs : str or None
        If None, all cses, else pass a string name for a specific stimuli type.
    hmm_engaged : bool
        If True, only include engaged trials.
    combine_pavlovian : bool
        If True, combine pavlovian trials with plus trials.
    across_run : bool
        If true, calculate the engagement combining across runs

    Returns
    -------
    array of bool
        True == incorrect trial

    """
    t2p = run.trace2p()
    conds, codes = t2p.conditions()
    errs = t2p.errors()
    if hmm_engaged:
        engage = engaged(run, across_run)
        conds = conds[engage]
        errs = errs[engage]
    if cs is not None:
        if combine_pavlovian and cs == 'plus':
            conds[conds == codes['pavlovian']] = codes['plus']
        try:
            errs = errs[conds == codes[cs]]
        except KeyError:
            # No trials of the given type
            errs = np.array([], dtype=bool)
    return errs
