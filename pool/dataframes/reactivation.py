import numpy as np
import pandas as pd

from .. import config
from . import behavior as bdf

def events_df(runs, threshold=0.1, xmask=False):

    events_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()
        c2p = run.classify2p()
        for event_type in config.stimuli():
            events = c2p.events(
                event_type, threshold=threshold, traces=t2p, xmask=xmask)
            index = pd.MultiIndex.from_product(
                [[run.mouse], [run.date], [run.run], [event_type],
                 np.arange(len(events))],
                names=['mouse', 'date', 'run', 'event_type', 'event_idx'])
            events_list.append(pd.DataFrame({'frame': events}, index=index))

    return pd.concat(events_list, axis=0)


def trial_events_df(
        runs, threshold=0.1, xmask=False, next_onset_pad_s=0.1,
        prev_onset_pad_s=2.5):

    result = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()

        all_onsets = t2p.csonsets()
        conditions = t2p.conditions()
        errors = t2p.errors(cs=None)

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate
        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        events = events_df([run], threshold, xmask=xmask)

        for trial_idx, (onset, next_onset, prev_onset, cond, err) in enumerate(zip(
                all_onsets, next_onsets, prev_onsets, conditions, errors)):

            trial_events = events.loc[
                (events.frame >= (prev_onset + prev_onset_pad_fr)) &
                (events.frame < (next_onset - next_onset_pad_fr))].copy()
            trial_events -= onset
            trial_events['time'] = trial_events.frame / fr

            # add in trial_idx, condition, error
            trial_events = pd.concat(
                [trial_events], keys=[trial_idx], names=['trial_idx'])
            trial_events = pd.concat(
                [trial_events], keys=[cond], names=['condition'])
            trial_events = pd.concat(
                [trial_events], keys=[err], names=['error'])

            result.append(trial_events)

    result_df = pd.concat(result, axis=0)
    result_df = result_df.reorder_levels(
        ['mouse', 'date', 'run', 'trial_idx', 'condition', 'error',
         'event_type', 'event_idx'])
    result_df.drop(columns=['frame'], inplace=True)

    return result_df


def peri_event_behavior_df(runs, threshold=0.1):

    behavior = bdf.behavior_df(runs)
    events = trial_events_df(
        runs, threshold=threshold, xmask=False)
    print(events.shape)
    edges = [-5, -0.1, 0, 2, 2.5, 5, 10]
    bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']
    events['time_cat'] = pd.cut(
        events.time, edges, labels=bin_labels)
    iti_events = events[events.time_cat == 'iti']
    print(iti_events.shape)

    result = [pd.DataFrame()]
    for event in iti_events.itertuples():
        mouse, date, run, trial_idx, condition, error, event_type, event_idx = \
            event.Index

        trial_errors = (behavior[behavior['condition'] == event_type]
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
            [prev_errors], keys=[event_idx], names=['event_idx'])

        next_errors = (trial_errors[trial_errors['trial_idx'] > trial_idx]
                       .iloc[:2])
        next_errors['trial_idx'] = np.arange(1, next_errors.shape[0] + 1)
        next_errors = pd.concat(
            [next_errors], keys=[condition], names=['condition'])
        next_errors = pd.concat(
            [next_errors], keys=[event_idx], names=['event_idx'])

        result.append(prev_errors)
        result.append(next_errors)

    result_df = pd.concat(result, axis=0)
    result_df = result_df.reorder_levels(
        ['mouse', 'date', 'run', 'condition', 'event_idx'])

    return result_df
