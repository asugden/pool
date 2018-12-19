import numpy as np
import pandas as pd

from . import reactivation

def behavior_df(runs):
    """Build a dataframe of all behavior results.

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

        conditions = t2p.conditions()
        # Generalize outcomes to it works for all stims?
        # outcomes = t2p.outcomes()
        errors = t2p.errors()

        index = pd.MultiIndex.from_product([
            [run.mouse], [run.date], [run.run], np.arange(len(conditions))],
            names=['mouse', 'date', 'run', 'trial_idx'])

        result_list.append(pd.DataFrame(
            {'conditions': conditions, 'errors': errors}, index=index))

    result = pd.concat(result_list, axis=0)

    return result


def peri_event_beahvior_df(runs, threshold=0.1):

    behavior = behavior_df(runs)
    events = reactivation.trial_events_df(
        runs, threshold=threshold, xmask=False)
    edges = [-5, -0.1, 0, 2, 2.5, 5, 10]
    bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']
    events['time_cat'] = pd.cut(
        events.time, edges, labels=bin_labels)
    events = events[events.time_cat.isin(['pre', 'post', 'iti'])]

    iti_events = events[events.time_cat == 'iti']
    print(iti_events.shape)

    result = [pd.DataFrame()]
    for event in iti_events.itertuples():
        mouse, date, run, trial_idx, condition, error, event_type, event_idx = \
            event.Index

        trial_errors = (behavior[behavior['conditions'] == event_type]
                        .loc[(mouse, date, run, slice(None)), 'errors']
                        .reset_index('trial_idx'))

        prev_errors = (trial_errors[trial_errors['trial_idx'] <= trial_idx]
                       .iloc[-2:]
                       .assign(trial_idx=[-2, -1]))

        # Put 'condition' and 'event_idx' back in the dataframe
        prev_errors = pd.concat(
            [prev_errors], keys=[condition], names=['conditions'])
        prev_errors = pd.concat(
            [prev_errors], keys=[event_idx], names=['event_idx'])

        next_errors = (trial_errors[trial_errors['trial_idx'] > trial_idx]
                       .iloc[:2]
                       .assign(trial_idx=[1, 2]))
        next_errors = pd.concat(
            [next_errors], keys=[condition], names=['conditions'])
        next_errors = pd.concat(
            [next_errors], keys=[event_idx], names=['event_idx'])

        result.append(prev_errors)
        result.append(next_errors)

    result_df = pd.concat(result, axis=0)
    result_df = result_df.reorder_levels(
        ['mouse', 'date', 'run', 'conditions', 'event_idx'])

    return result_df
