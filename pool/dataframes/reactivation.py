"""All functions return reactivation-related dataframes."""

import numpy as np
import pandas as pd

from .. import config
from .. import database
from . import behavior as bdf


def events_df(runs, threshold=0.1, xmask=False, inactivity_mask=False):
    """
    Return all event frames.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    threshold : float
        Classifier cutoff probability.
    xmask : bool
        If True, only allow one event (across types) per time bin.
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, event_idx, event_type
        Columns : frame

    """
    events_list = [pd.DataFrame({})]
    for run in runs:
        t2p = run.trace2p()
        c2p = run.classify2p()
        if inactivity_mask:
            mask = t2p.inactivity()
        else:
            mask = None
        # Keep track of frames/events so that we can give them an overall idx
        frame_event_tuples = []
        for event_type in config.stimuli():
            events = c2p.events(
                event_type, threshold=threshold, traces=t2p, xmask=xmask,
                mask=mask)
            frame_event_tuples.extend(
                [(event, event_type) for event in events])
        sorted_frame_event_tuples = sorted(frame_event_tuples)
        if len(sorted_frame_event_tuples):
            # Un-zip the (frame, event_type) tuples
            run_events, run_event_types = zip(*sorted_frame_event_tuples)
        else:
            run_events, run_event_types = [], []
        index = pd.MultiIndex.from_arrays(
            [[run.mouse] * len(run_events), [run.date] * len(run_events),
             [run.run] * len(run_events), range(len(run_events)),
             run_event_types],
            names=['mouse', 'date', 'run', 'event_idx', 'event_type'])
        events_list.append(
            pd.DataFrame({'frame': run_events}, index=index))

    return pd.concat(events_list, axis=0)


def trial_classifier_df(runs):
    """
    Return classifier probability across trials.

    Parameters
    ----------
    runs : RunSorter or list of Runs

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, condition, error, time
        Columns : [one per replay type, i.e. 'plus', 'neutral', 'minus']

    """
    result = [pd.DataFrame()]
    db = database.db()
    for run in runs:
        result.append(db.get(
            'trialdf_classifier', mouse=run.mouse, date=run.date, run=run.run,
            metadata_object=run))
    result = pd.concat(result, axis=0)

    return result


def trial_events_df(
        runs, threshold=0.1, xmask=False, inactivity_mask=False):
    """
    Return reactivation events relative to stimuli presentations.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    threshold : float
        Classifier cutoff probability.
    xmask : bool
        If True, only allow one event (across types) per time bin.
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.

    Note
    ----
    Events are included multiple times, bot before the next stim and after the
    previous stim presentation!

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, condition, error, event_type, event_idx
        Columns : time

    """
    result = [pd.DataFrame()]
    db = database.db()
    analysis = 'trialdf_events_{}_{}_{}'.format(
        threshold,
        'xmask' if xmask else 'noxmask',
        'inactmask' if inactivity_mask else 'noinactmask')
    for run in runs:
        result.append(db.get(
            analysis, mouse=run.mouse, date=run.date, run=run.run,
            metadata_object=run))
    result = pd.concat(result, axis=0)

    return result


def peri_event_behavior_df(runs, threshold=0.1):

    behavior = bdf.behavior_df(runs)
    events = trial_events_df(
        runs, threshold=threshold, xmask=False)
    edges = [-5, -0.1, 0, 2, 2.5, 5, 10]
    bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']
    events['time_cat'] = pd.cut(
        events.time, edges, labels=bin_labels)
    iti_events = events[events.time_cat == 'iti']

    result = [pd.DataFrame()]
    for event in iti_events.itertuples():
        mouse, date, run, trial_idx, condition, error, event_type, event_idx = \
            event.Index

        for condition in behavior['condition'].unique():
            trial_errors = (behavior[behavior['condition'] == condition]
                            .loc[(mouse, date, run, slice(None)), 'error']
                            .reset_index('trial_idx'))

            prev_errors = (trial_errors[trial_errors['trial_idx'] <= trial_idx]
                           .iloc[-2:])
            # Reset trial_idx to be relative to event, handling edge cases
            prev_errors['trial_idx'] = np.arange(-prev_errors.shape[0], 0)
            # Put 'condition' and 'event_idx' back in the dataframe
            prev_errors = pd.concat(
                [prev_errors], keys=[condition], names=['condition'])
            prev_errors = pd.concat(
                [prev_errors], keys=[event_type], names=['event_type'])
            prev_errors = pd.concat(
                [prev_errors], keys=[event_idx], names=['event_idx'])

            next_errors = (trial_errors[trial_errors['trial_idx'] > trial_idx]
                           .iloc[:2])
            next_errors['trial_idx'] = np.arange(1, next_errors.shape[0] + 1)
            next_errors = pd.concat(
                [next_errors], keys=[condition], names=['condition'])
            next_errors = pd.concat(
                [next_errors], keys=[event_type], names=['event_type'])
            next_errors = pd.concat(
                [next_errors], keys=[event_idx], names=['event_idx'])

            result.append(prev_errors)
            result.append(next_errors)

    result_df = pd.concat(result, axis=0)
    result_df = result_df.reorder_levels(
        ['mouse', 'date', 'run', 'condition', 'event_type', 'event_idx'])

    return result_df
