import numpy as np
import pandas as pd

from .. import config
from . import behavior as bdf


def events_df(runs, threshold=0.1, xmask=False, use_inactivity_mask=False):
    events_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()
        c2p = run.classify2p()
        if use_inactivity_mask:
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

def trial_classifier_df(runs, errortrials=-1, next_onset_pad_s=0.1,
        prev_onset_pad_s=2.5)
    result = [pd.DataFrame()]
    for run in runs:
        c2p = run.classify2p()
        t2p = run.trace2p()

        classifier_results = c2p.results()
        all_onsets = t2p.csonsets()
        conditions = t2p.conditions()
        errors = t2p.errors(cs=None)
        replay_types = config.stimuli()

        next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
        prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

        fr = t2p.framerate

        next_onset_pad_fr = int(np.ceil(next_onset_pad_s * fr))
        prev_onset_pad_fr = int(np.ceil(prev_onset_pad_s * fr))

        for trial_idx, (onset, next_onset, prev_onset, cond, err) in enumerate(
                zip(all_onsets, next_onsets, prev_onsets, conditions, errors)):
            if errortrials == 0 and err:
                continue
            elif errortrials == 1 and not err:
                continue

            start_fr = prev_onset + prev_onset_pad_fr
            end_fr = next_onset - next_onset_pad_fr
            pre_fr = onset - start_fr

            trial_result = []
            for replay_type in replay_types:
                trial_replay_result = classifier_results[replay_type][
                    start_fr:end_fr - 1]
                time = np.arange(len(trial_replay_result)) / fr - pre_fr

                index = pd.MultiIndex.from_product(
                    [[trial_idx], [cond], time],
                    names=['trial_idx', 'condition', 'time'])
                trial_result.append(
                    pd.Series(trial_replay_result, index=index, name=replay_type))
            result.append(pd.concat(trial_result, axis=1))

    final_result = pd.concat(result, axis=0)

    return final_result

def trial_events_df(
        runs, threshold=0.1, xmask=False, next_onset_pad_s=0.1,
        prev_onset_pad_s=2.5, use_inactivity_mask=False):

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

        events = events_df(
            [run], threshold, xmask=xmask,
            use_inactivity_mask=use_inactivity_mask)

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
