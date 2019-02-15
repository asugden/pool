"""All functions return reactivation-related DataFrames."""
from builtins import range, zip

import numpy as np
import pandas as pd

from .. import config
from .. import database
from . import behavior as bdf
from ..calc import aligned_dfs

POST_PAD_S = 0.3
POST_PAVLOVIAN_PAD_S = 0.6
PRE_PAD_S = 0.2


def events_df(
        runs, threshold=0.1, xmask=False, inactivity_mask=False,
        stimulus_mask=False):
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
    stimulus_mask : bool
        If True, remove all events during stimulus presentation.

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
            inact_mask = t2p.inactivity()
        if stimulus_mask:
            all_stim_mask = t2p.trialmask(
                cs='', errortrials=-1, fulltrial=False, padpre=PRE_PAD_S,
                padpost=POST_PAD_S)
            pav_mask = t2p.trialmask(
                cs='pavlovian', errortrials=-1, fulltrial=False,
                padpre=PRE_PAD_S, padpost=POST_PAVLOVIAN_PAD_S)
            blank_mask = t2p.trialmask(
                cs='blank', errortrials=-1, fulltrial=False,
                padpre=PRE_PAD_S, padpost=POST_PAVLOVIAN_PAD_S)
            stim_mask = np.invert((all_stim_mask | pav_mask) & (~blank_mask))
            # stim_mask = np.invert(all_stim_mask | pav_stim_mask)
        if inactivity_mask and stimulus_mask:
            mask = inact_mask & stim_mask
        elif inactivity_mask:
            mask = inact_mask
        elif stimulus_mask:
            mask = stim_mask
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
            run_events, run_event_types = list(zip(*sorted_frame_event_tuples))
        else:
            run_events, run_event_types = [], []
        index = pd.MultiIndex.from_arrays(
            [[run.mouse] * len(run_events), [run.date] * len(run_events),
             [run.run] * len(run_events), range(len(run_events))],
            names=['mouse', 'date', 'run', 'event_idx'])
        events_list.append(
            pd.DataFrame({'frame': run_events, 'event_type': run_event_types},
                         index=index))

    return pd.concat(events_list, axis=0)


def trial_classifier_df(runs, pad_s=None):
    """
    Return classifier probability across trials.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, time
        Columns {replay_type}

    """
    result = [pd.DataFrame()]
    for run in runs:
        result.append(aligned_dfs.trial_classifier_probability(
            run, pad_s=pad_s))
    result = pd.concat(result, axis=0)

    return result


def trial_events_df(
        runs, threshold=0.1, xmask=False, inactivity_mask=False, pad_s=None):
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
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus. Be careful changing this and make sure it
        matched trial_frames_df if used together.

    Note
    ----
    Events are included multiple times, both before the next stim and after the
    previous stim presentation!

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, event_idx
        Columns : event_type, abs_frame, time, trial_idx

    """
    result = [pd.DataFrame()]
    for run in runs:
        result.append(aligned_dfs.trial_events(
            run, threshold=threshold, xmask=xmask,
            inactivity_mask=inactivity_mask, pad_s=pad_s))
    result = pd.concat(result, axis=0)

    return result


def trigger_events_df_orig(
        runs, trigger, threshold=0.1, xmask=False, inactivity_mask=False):
    """Return reactivation events aligned to various triggers."""
    result = [pd.DataFrame()]
    db = database.db()
    analysis = 'trialdf_{}_events_{}_{}_{}'.format(
        trigger,
        threshold,
        'xmask' if xmask else 'noxmask',
        'inactmask' if inactivity_mask else 'noinactmask')
    for run in runs:
        result.append(db.get(
            analysis, mouse=run.mouse, date=run.date, run=run.run,
            metadata_object=run))
    result = pd.concat(result, axis=0)

    return result


def trigger_events_df(
        runs, trigger, pre_s=-1., post_s=5., threshold=0.1, xmask=False,
        inactivity_mask=False, stimulus_mask=True):
    """
    Determine event times aligned to other (other than stimulus) events.

    Parameters
    ----------
    runs : RunSorter
    trigger : {'punishment', 'reward', 'lickbout'}
        Event to trigger PSTH on.
    pre_s, post_s : float
        Time around each event to look.
    threshold : float
        Classifier cutoff probability.
    xmask : bool
        If True, only allow one event (across types) per time bin.
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.
    stimulus_mask : bool
        If True, remove all events during stimulus presentation.

    Note
    ----
    Individual events may appear in this DataFrame multiple times!
    Events may show up both as being after a triggering event and before
    the next one.

    """
    # Initialize with an empty DataFrame that will match same format as output
    result = []
    for run in runs:
        t2p = run.trace2p()

        if trigger == 'reward':
            onsets = t2p.reward()
            # There are 0s in place of un-rewarded trials.
            onsets = onsets[onsets > 0]
        elif trigger == 'punishment':
            onsets = t2p.punishment()
            # There are 0s in place of un-punished trials.
            onsets = onsets[onsets > 0]
        elif trigger == 'lickbout':
            onsets = t2p.lickbout()
        else:
            raise ValueError("Unrecognized 'trigger' value.")

        fr = t2p.framerate
        pre_fr = int(np.ceil(-pre_s * fr))
        post_fr = int(np.ceil(post_s * fr))

        events = events_df(
            [run], threshold, xmask=xmask,
            inactivity_mask=inactivity_mask,
            stimulus_mask=stimulus_mask)

        for trigger_idx, onset in enumerate(onsets):

            trigger_events = events.loc[
                (events.frame >= (onset - pre_fr)) &
                (events.frame < (onset + post_fr))].copy()
            trigger_events['time'] = (trigger_events.frame - onset) / fr
            trigger_events['trigger_idx'] = trigger_idx

            result.append(trigger_events)

    if len(result):
        result_df = (pd
                     .concat(result, axis=0)
                     .rename(columns={'frame': 'abs_frame'})
                     .sort_index()
                     )
    else:
        result_df = pd.DataFrame(
            {'trigger_idx': [], 'event_type': [], 'abs_frame': [], 'time': []},
            index=pd.MultiIndex(
                levels=[[], [], [], []],
                labels=[[], [], [], []],
                names=['mouse', 'date', 'run', 'event_idx']))

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
        mouse, date, run, event_idx = event.Index

        for condition in behavior['condition'].unique():
            trial_errors = (behavior[behavior['condition'] == event.condition]
                            .loc[(mouse, date, run, slice(None)), 'error']
                            .reset_index('trial_idx'))

            prev_errors = (trial_errors[trial_errors[
                                        'trial_idx'] <= event.trial_idx]
                           .iloc[-2:])
            # Reset trial_idx to be relative to event, handling edge cases
            prev_errors['trial_idx'] = np.arange(-prev_errors.shape[0], 0)
            # Put 'condition' and 'event_idx' back in the dataframe
            prev_errors = pd.concat(
                [prev_errors], keys=[event.condition], names=['condition'])
            prev_errors = pd.concat(
                [prev_errors], keys=[event.event_type], names=['event_type'])
            prev_errors = pd.concat(
                [prev_errors], keys=[event_idx], names=['event_idx'])

            next_errors = (trial_errors[trial_errors[
                                        'trial_idx'] > event.trial_idx]
                           .iloc[:2])
            next_errors['trial_idx'] = np.arange(1, next_errors.shape[0] + 1)
            next_errors = pd.concat(
                [next_errors], keys=[event.condition], names=['condition'])
            next_errors = pd.concat(
                [next_errors], keys=[event.event_type], names=['event_type'])
            next_errors = pd.concat(
                [next_errors], keys=[event_idx], names=['event_idx'])

            result.append(prev_errors)
            result.append(next_errors)

    result_df = pd.concat(result, axis=0)
    result_df = result_df.reorder_levels(
        ['mouse', 'date', 'run', 'condition', 'event_type', 'event_idx'])

    return result_df
