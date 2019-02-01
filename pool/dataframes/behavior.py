import numpy as np
import pandas as pd

from .. import config
from .. import database
from ..calc import behavior


def behavior_df(runs):
    """
    Build a DataFrame of all behavior results.

    Parameters
    ----------
    runs : RunSorter or list of Runs

    Returns
    -------
    pd.DataFrame

    """
    result_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()

        conditions = t2p.conditions(return_as_strings=True)
        # Generalize outcomes to it works for all stims?
        # outcomes = t2p.outcomes()
        errors = t2p.errors()

        index = pd.MultiIndex.from_product([
            [run.mouse], [run.date], [run.run], np.arange(len(conditions))],
            names=['mouse', 'date', 'run', 'trial_idx'])

        result_list.append(pd.DataFrame(
            {'condition': conditions, 'error': errors}, index=index))

    result = pd.concat(result_list, axis=0)

    return result


def behavior_metric_df(dates, hmm_engaged=True):
    """
    Build a DataFrame of performance per-day.

    Parameters
    ----------
    dates : DateSorter or list of Dates
    engaged : boolean
        If True, use HMM engagement model to limit to engaged trials.

    Result
    ------
    pd.DataFrame

    """
    result_list = [pd.DataFrame()]
    for date in dates:
        data = {}
        data['dprime'] = behavior.dprime(
            date, hmm_engaged=hmm_engaged, combine_pavlovian=False,
            combine_passives=True)
        data['correct_fraction'] = behavior.correct_fraction(
            date, cs=None, hmm_engaged=hmm_engaged, combine_pavlovian=False)
        for stim in config.stimuli():
            data['correct_fraction_{}'.format(stim)] = \
                behavior.correct_fraction(
                    date, cs=stim, hmm_engaged=hmm_engaged,
                    combine_pavlovian=False)
        data['criterion'] = behavior.criterion(
            date, hmm_engaged=hmm_engaged, combine_pavlovian=False,
            combine_passives=True)
        data['likelihood_ratio'] = behavior.likelihood_ratio(
            date, hmm_engaged=hmm_engaged, combine_pavlovian=False,
            combine_passives=True)

        index = pd.MultiIndex.from_tuples(
            [(date.mouse, date.date)], names=['mouse', 'date'])
        result_list.append(
            pd.DataFrame(data, index=index))

    return pd.concat(result_list, axis=0)
