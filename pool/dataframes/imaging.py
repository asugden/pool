from __future__ import division

import numpy as np
import pandas as pd

from .. import database
from ..calc import aligned_dfs

PRE_S = 5
POST_S = 5

POST_PAD_S = 2.3
POST_PAVLOVIAN_PAD_S = 2.6
PRE_PAD_S = 0.2


def frames_df(runs, inactivity_mask=False, stimulus_mask=False):
    """
    Return all frame times.

    Parameters
    ----------
    runs : RunSorter
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.

    Returns
    -------
    pd.DataFrame

    """
    frames_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()
        frame_period = 1. / t2p.framerate
        frames = np.arange(t2p.nframes)

        if inactivity_mask:
            inact_mask = t2p.inactivity()
        if stimulus_mask:
            all_stim_mask = t2p.trialmask(
                cs='', errortrials=-1, fulltrial=False, padpre=PRE_PAD_S,
                padpost=POST_PAD_S)
            pav_stim_mask = t2p.trialmask(
                cs='pavlovian', errortrials=-1, fulltrial=False,
                padpre=PRE_PAD_S, padpost=POST_PAVLOVIAN_PAD_S)
            stim_mask = np.invert(all_stim_mask | pav_stim_mask)
        if inactivity_mask and stimulus_mask:
            frames = frames[inact_mask & stim_mask]
        elif inactivity_mask:
            frames = frames[inact_mask]
        elif stimulus_mask:
            frames = frames[stim_mask]

        index = pd.MultiIndex.from_product(
            [[run.mouse], [run.date], [run.run], frames],
            names=['mouse', 'date', 'run', 'frame'])
        frames_list.append(pd.DataFrame(
            {'frame_period': frame_period}, index=index))

    return pd.concat(frames_list, axis=0)


def trial_frames_df(runs, inactivity_mask=False, pad_s=None):
    """
    Return acquisition frames relative to stimuli presentations.

    Parameters
    ----------
    runs : RunSorter
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.
    pad_s : 2-element tuple of float
        Used to calculate the padded end of the previous stimulus and the
        padded start of the next stimulus when cutting up output. Does NOT
        pad the current stimulus. Be careful changing this and make sure it
        matched trial_events_df if used together.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx
        Columns : frame, frame_period, time

    """
    result = [pd.DataFrame()]
    for run in runs:
        result.append(aligned_dfs.trial_frames(
            run, inactivity_mask=inactivity_mask, pad_s=pad_s))
    result = pd.concat(result, axis=0)

    return result


def trigger_frames_df(runs, trigger, inactivity_mask=False):
    """
    Calculate a trigger-aligned imaging DataFrame.

    Similar to trial_frames_df, but with non-trial triggers.

    Parameters
    ----------
    runs : RunSorter
    trigger : {'reward', 'punishment', 'lickbout'}
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.

    Returns
    -------
    pd.DataFrame

    """
    result = [pd.DataFrame()]
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

        fr = t2p.framerate
        pre_fr = PRE_S * fr
        post_fr = POST_S * fr

        frames = (frames_df([run], inactivity_mask, stimulus_mask=True)
                  .reset_index(['frame'])
                  )

        for trigger_idx, onset in enumerate(onsets):

            trigger_frames = frames.loc[
                (frames.frame > (onset - pre_fr)) &
                (frames.frame < (onset + post_fr))].copy()
            trigger_frames['frame'] -= onset
            trigger_frames['time'] = trigger_frames.frame / fr
            trigger_frames['trigger_idx'] = trigger_idx

            result.append(trigger_frames)

    result_df = (pd
                 .concat(result, axis=0)
                 .set_index(['trigger_idx', 'frame'], append=True)
                 )

    return result_df


def trial_stimulus_response_df(dates):
    """
    Calculate the response to stimuli for each cell per trial.

    Parameters
    ----------
    dates : DateSorter

    Returns
    -------
    pd.DataFrame

    """
    result = [pd.DataFrame()]
    db = database.db()
    analysis = 'stim_dff_alltrials_pertrial'
    for date in dates:
        result.append(db.get(
            analysis, mouse=date.mouse, date=date.date,
            metadata_object=date, force=False))
    result = pd.concat(result, axis=0)

    return result
