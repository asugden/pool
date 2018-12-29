"""All functions return reactivation-related dataframes."""

import numpy as np
import pandas as pd

from .. import config
from .. import database
from .dataframes import smart_merge
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
        Index : mouse, date, run, event_type, event_idx
        Columns : frame

    """
    events_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()
        c2p = run.classify2p()
        if inactivity_mask:
            mask = t2p.inactivity()
        else:
            mask = None
        for event_type in config.stimuli():
            events = c2p.events(
                event_type, threshold=threshold, traces=t2p, xmask=xmask,
                mask=mask)
            index = pd.MultiIndex.from_product(
                [[run.mouse], [run.date], [run.run], [event_type],
                 np.arange(len(events))],
                names=['mouse', 'date', 'run', 'event_type', 'event_idx'])
            events_list.append(pd.DataFrame({'frame': events}, index=index))

    return pd.concat(events_list, axis=0)


def trial_classifier_df_orig(
        runs, prev_onset_pad_s=2.5, next_onset_pad_s=0.1):
    """
    Return classifier probability across trials.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    prev_onset_pad_s : float
        Pre-trial windows starts this long after last trial onset.
    next_offset_pad_s : float
        Post-trial ITI includes up to this long before the next trial onset.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, condition, time
        Columns : [one per replay type, i.e. 'plus', 'neutral', 'minus']

    """
    result = [pd.DataFrame()]
    for run in runs:
        c2p = run.classify2p()
        t2p = run.trace2p()

        classifier_results = c2p.results()
        all_onsets = t2p.csonsets()
        conditions = t2p.conditions()
        replay_types = config.stimuli()

        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)
        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)

        fr = t2p.framerate

        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        for trial_idx, (onset, next_onset, prev_onset, cond) in enumerate(
                zip(all_onsets, next_onsets, prev_onsets, conditions)):

            start_fr = prev_onset + prev_onset_pad_fr
            end_fr = next_onset - next_onset_pad_fr
            pre_fr = onset - start_fr

            trial_result = [pd.DataFrame()]
            for replay_type in replay_types:
                trial_replay_result = classifier_results[replay_type][
                    start_fr:end_fr - 1]
                time = (np.arange(len(trial_replay_result)) - pre_fr) / fr

                index = pd.MultiIndex.from_product(
                    [[run.mouse], [run.date], [run.run], [trial_idx], [cond],
                     time],
                    names=['mouse', 'date', 'run', 'trial_idx', 'condition',
                           'time'])
                trial_result.append(
                    pd.Series(trial_replay_result, index=index,
                              name=replay_type))
            result.append(pd.concat(trial_result, axis=1))

    final_result = pd.concat(result, axis=0)

    return final_result


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


# def trial_events_df_orig(
#         runs, threshold=0.1, xmask=False, prev_onset_pad_s=2.5,
#         next_onset_pad_s=0.1, inactivity_mask=False):

#     result = [pd.DataFrame()]
#     for run in runs:
#         t2p = run.trace2p()

#         all_onsets = t2p.csonsets()
#         conditions = t2p.conditions()
#         errors = t2p.errors(cs=None)

#         next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
#         prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

#         fr = t2p.framerate
#         next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
#         prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

#         events = events_df(
#             [run], threshold, xmask=xmask,
#             inactivity_mask=inactivity_mask)

#         for trial_idx, (onset, next_onset, prev_onset, cond, err) in enumerate(zip(
#                 all_onsets, next_onsets, prev_onsets, conditions, errors)):

#             trial_events = events.loc[
#                 (events.frame >= (prev_onset + prev_onset_pad_fr)) &
#                 (events.frame < (next_onset - next_onset_pad_fr))].copy()
#             trial_events -= onset
#             trial_events['time'] = trial_events.frame / fr

#             # add in trial_idx, condition, error
#             trial_events = pd.concat(
#                 [trial_events], keys=[trial_idx], names=['trial_idx'])
#             trial_events = pd.concat(
#                 [trial_events], keys=[cond], names=['condition'])
#             trial_events = pd.concat(
#                 [trial_events], keys=[err], names=['error'])

#             result.append(trial_events)

#     result_df = pd.concat(result, axis=0)
#     result_df = result_df.reorder_levels(
#         ['mouse', 'date', 'run', 'trial_idx', 'condition', 'error',
#          'event_type', 'event_idx'])
#     result_df.drop(columns=['frame'], inplace=True)

#     return result_df


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
