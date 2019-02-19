"""Aligned DataFrames."""
import numpy as np
import pandas as pd

import pool
from .. import config
from ..database import memoize

POST_PAD_S = 2.3
POST_PAVLOVIAN_PAD_S = 2.6
PRE_PAD_S = 0.2


@memoize(across='run', updated=190213, large_output=True)
def trial_classifier_probability(run, pad_s=None):
    """
    Return the classifier probability aligned to trial onsets.

    Parameters
    ----------
    run : Run
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, time
        Columns : {replay_type}

    """
    if pad_s is None:
        pad_s = (PRE_PAD_S, POST_PAD_S)

    c2p = run.classify2p()
    t2p = run.trace2p()

    classifier_results = c2p.results()
    all_onsets = t2p.csonsets()
    replay_types = config.stimuli()

    prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)
    next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)

    fr = t2p.framerate

    pre_onset_pad_fr = int(np.ceil(pad_s[0] * fr))
    post_onset_pad_fr = int(np.ceil(pad_s[1] * fr))

    result = [pd.DataFrame()]
    for trial_idx, (onset, next_onset, prev_onset) in enumerate(
            zip(all_onsets, next_onsets, prev_onsets)):

        start_fr = prev_onset + post_onset_pad_fr
        end_fr = next_onset - pre_onset_pad_fr
        pre_fr = onset - start_fr

        trial_result = [pd.DataFrame()]
        for replay_type in replay_types:
            trial_replay_result = classifier_results[replay_type][
                start_fr:end_fr - 1]
            time = (np.arange(len(trial_replay_result)) - pre_fr) / fr

            index = pd.MultiIndex.from_product(
                [[run.mouse], [run.date], [run.run], [trial_idx], time],
                names=['mouse', 'date', 'run', 'trial_idx', 'time'])
            trial_result.append(
                pd.Series(trial_replay_result, index=index,
                          name=replay_type))
        result.append(pd.concat(trial_result, axis=1))
    final_result = pd.concat(result, axis=0)

    return final_result


@memoize(across='run', updated=190215, large_output=True)
def trial_events(
        run, threshold=0.1, xmask=False, inactivity_mask=False, pad_s=None):
    """
    Determine event times relative to stimulus onset.

    Parameters
    ----------
    run : Run
    threshold : float
        Classifier cutoff probability.
    xmask : bool
        If True, only allow one event (across types) per time bin.
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus.

    Note
    ----
    Individual events will appear in this DataFrame multiple times!
    Events will show up both as being after a stimulus and before
    the next one.

    """
    if pad_s is None:
        pad_s = (PRE_PAD_S, POST_PAD_S)

    t2p = run.trace2p()

    all_onsets = t2p.csonsets()

    next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
    prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

    fr = t2p.framerate
    pre_onset_pad_fr = int(np.ceil(pad_s[0] * fr))
    post_onset_pad_fr = int(np.ceil(pad_s[1] * fr))

    events = pool.dataframes.reactivation.events_df(
        [run], threshold, xmask=xmask,
        inactivity_mask=inactivity_mask)

    result = [pd.DataFrame()]
    for trial_idx, (onset, next_onset, prev_onset) in enumerate(zip(
            all_onsets, next_onsets, prev_onsets)):

        trial_events = events.loc[
            (events.frame >= (prev_onset + post_onset_pad_fr)) &
            (events.frame < (next_onset - pre_onset_pad_fr))].copy()
        trial_events['time'] = (trial_events.frame - onset) / fr
        trial_events['trial_idx'] = trial_idx

        result.append(trial_events)

    result_df = (pd
                 .concat(result, axis=0)
                 .rename(columns={'frame': 'abs_frame'})
                 .sort_index()
                 )

    return result_df


@memoize(across='run', updated=190215, large_output=True)
def trial_frames(run, inactivity_mask=False, pad_s=None):
    """
    Return acquisition frames relative to stimuli presentations.

    Parameters
    ----------
    run : Run
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx
        Columns : frame, frame_period, time

    """
    if pad_s is None:
        pad_s = (PRE_PAD_S, POST_PAD_S)

    t2p = run.trace2p()

    all_onsets = t2p.csonsets()

    next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
    prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

    fr = t2p.framerate
    pre_onset_pad_fr = int(np.ceil(pad_s[0] * fr))
    post_onset_pad_fr = int(np.ceil(pad_s[1] * fr))

    frames = (pool.dataframes.imaging
              .frames_df([run], inactivity_mask)
              .reset_index(['frame'])
              )

    result = [pd.DataFrame()]
    for trial_idx, (onset, next_onset, prev_onset) in \
            enumerate(zip(all_onsets, next_onsets, prev_onsets)):
        # Pull out events around the current onset
        trial_frames = frames.loc[
            (frames.frame >= (prev_onset + post_onset_pad_fr)) &
            (frames.frame < (next_onset - pre_onset_pad_fr))].copy()

        # Convert to relative times
        trial_frames.frame -= onset

        # Add in some additional information
        trial_frames['time'] = \
            trial_frames.frame * trial_frames.frame_period
        trial_frames['trial_idx'] = trial_idx

        result.append(trial_frames)

    result_df = (pd
                 .concat(result, axis=0)
                 .set_index(['trial_idx', 'frame'], append=True)
                 )

    return result_df
