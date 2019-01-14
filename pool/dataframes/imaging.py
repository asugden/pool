import numpy as np
import pandas as pd

from .. import database


def frames_df(runs, inactivity_mask=False):
    frames_list = [pd.DataFrame()]
    for run in runs:
        t2p = run.trace2p()
        frame_period = 1. / t2p.framerate
        frames = np.arange(t2p.nframes)
        if inactivity_mask:
            frames = frames[t2p.inactivity()]
        index = pd.MultiIndex.from_product(
            [[run.mouse], [run.date], [run.run], frames],
            names=['mouse', 'date', 'run', 'frame'])
        frames_list.append(pd.DataFrame(
            {'frame_period': frame_period}, index=index))

    return pd.concat(frames_list, axis=0)


# def trial_frames_df_orig(runs, next_onset_pad_s=0.1, prev_onset_pad_s=2.5):
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

#         frames = frames_df(run)

#         for trial_idx, (onset, next_onset, prev_onset, cond, err) in \
#                 enumerate(zip(all_onsets, next_onsets, prev_onsets, conditions,
#                               errors)):

#             trial_frames = frames.loc[
#                 (frames.frame >= prev_onset + prev_onset_pad_fr) &
#                 (frames.frame < next_onset - next_onset_pad_fr)].copy()
#             trial_frames.frame -= onset
#             trial_frames['time'] = \
#                 trial_frames.frame * trial_frames.frame_period

#             # add in trial_idx
#             trial_frames = pd.concat(
#                 [trial_frames], keys=[trial_idx], names=['trial_idx'])

#             result.append(trial_frames)

#     result_df = pd.concat(result, axis=0)
#     result_df = result_df.reorder_levels(
#         ['mouse', 'date', 'run', 'trial_idx'])
#     result_df.drop(columns=['frame'], inplace=True)

#     return result_df


def trial_frames_df(runs, inactivity_mask=False):
    """
    Return acquisition frames relative to stimuli presentations.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    inactivity_mask : bool
        If True, enforce that all events are during times of inactivity.

    Returns
    -------
    pd.DataFrame
        Index : mouse, date, run, trial_idx, condition, error
        Columns : frame, frame_period, time

    """
    result = [pd.DataFrame()]
    db = database.db()
    analysis = 'trialdf_frames_{}'.format(
        'inactmask' if inactivity_mask else 'noinactmask')
    for run in runs:
        result.append(db.get(
            analysis, mouse=run.mouse, date=run.date, run=run.run,
            metadata_object=run))
    result = pd.concat(result, axis=0)

    return result
