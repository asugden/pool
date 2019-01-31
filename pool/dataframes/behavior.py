import numpy as np
import pandas as pd

from .. import config
from .. import database


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


def behavior_metric_df(dates, engaged=True):
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
    if engaged:
        raise NotImplementedError
    result_list = [pd.DataFrame()]
    db = database.db()
    for date in dates:
        data = {}
        if engaged:
            pass
        else:
            data['dprime'] = db.get(
                'behavior_dprime_orig', mouse=date.mouse, date=date.date,
                metadata_object=date)
            data['behavior'] = db.get(
                'behavior_orig', mouse=date.mouse, date=date.date,
                metadata_object=date)
            data['LR'] = db.get(
                'behavior_LR_orig', mouse=date.mouse, date=date.date,
                metadata_object=date)
            data['criterion'] = db.get(
                'behavior_criterion_orig', mouse=date.mouse, date=date.date,
                metadata_object=date)
            for stim in config.stimuli():
                data['behavior_{}'.format(stim)] = db.get(
                    'behavior_{}_orig'.format(stim), mouse=date.mouse,
                    date=date.date, metadata_object=date)

        index = pd.MultiIndex.from_tuples(
            [(date.mouse, date.date)], names=['mouse', 'date'])
        result_list.append(
            pd.DataFrame(data, index=index))

    return pd.concat(result_list, axis=0)
