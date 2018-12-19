"""Reactivation plotting functions."""
import numpy as np
import pandas as pd

from .. import config


def reactivation_probability_throughout_trials(
        ax, runs, trial_type='plus', replay_type='plus',
        pre_s=2, post_s=None, errortrials=-1, **kwargs):
    """Plot the classifier probability throughout the trial.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes
    runs : RunSorter
    trial_type : str
        Type of trial to plot data from.
    replay_type : str
        Type of replay to plot classifier probabilities for.
    pre_s : float
        Time in seconds prior to stim onset.
    post_s : float, optional
        If not None, only include this duration after stimulus (in seconds).
    errortrials : {-1, 0, 1}
        -1 is all trials, 0 is correct trials, 1 is error trials

    """
    # Add in a few more things to kwargs
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = config.colors(replay_type)
    if 'label' not in kwargs:
        kwargs['label'] = replay_type

    all_results = [pd.DataFrame()]
    for run in runs:
        all_results.append(_classifier_by_trial(
            run, pre_s=pre_s, post_s=post_s,
            errortrials=errortrials))
    result = pd.concat(all_results, axis=0)

    grouped = result.groupby(level=['condition', 'time']).mean()

    # We could have no trials of the desired error type (or trial type in general)
    if trial_type in grouped.index:
        ax.plot(grouped.xs(trial_type).loc[:, replay_type], **kwargs)

    # def percentile10(x):
    #     return np.percentile(x, 0.1)
    # def percentile90(x):
    #     return np.percentile(x, 0.9)
    # grouped = result.groupby(level=['condition', 'time']).agg([np.median, percentile10, percentile90])
    # ax.fill_between(grouped.loc[trial_type, replay_type].index,
    #                 grouped.loc[trial_type, (replay_type, 'percentile10')],
    #                 grouped.loc[trial_type, (replay_type, 'percentile90')], alpha=0.5, **kwargs)


def reactivation_events_throughout_trials(
        ax, runs, pre_s=5, iti_start_s=5, iti_end_s=10, stim_pad_s=0.1,
        threshold=0.1):
    edges = [-pre_s, -stim_pad_s, 0, 2, 2 + stim_pad_s, iti_start_s, iti_end_s]
    bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']

    for run in runs:
        events, times = _events_by_trial(run, threshold, xmask=False)

        events_binned = _bin_events(events, times, edges, bin_labels)


def _classifier_by_trial(run, pre_s=2, post_s=None, errortrials=-1):
    """Return the classifier probability of all stimuli around stim presentations.

    Parameters
    ----------
    c2p : flow.classify2p.Classify2p
    t2p : flow.trace2p.Trace2p
    pre_s : float
        Time in seconds prior to stim onset.
    post_s : float, optional
        If not None, only include this duration after stimulus (in seconds).
    errortrials : {-1, 0, 1}
        -1 is all trials, 0 is correct trials, 1 is error trials

    """
    c2p = run.classify2p()
    t2p = run.trace2p()

    classifier_results = c2p.results()
    all_onsets = t2p.csonsets()
    conditions = t2p.conditions()
    errors = t2p.errors(cs=None)
    replay_types = config.stimuli()

    # Figure out last offset. Either last frame or largest stim onset diff,
    # whichever is smaller
    max_onset_diff = np.diff(all_onsets).max()
    last_offset = t2p.nframes if t2p.nframes - all_onsets[-1] <= max_onset_diff \
        else all_onsets[-1] + max_onset_diff

    next_onsets = np.concatenate([all_onsets[1:], last_offset], axis=None)

    fr = t2p.framerate
    pre_f = int(np.ceil(pre_s * fr))

    result = [pd.DataFrame()]
    for trial_idx, (onset, next_onset, cond, err) in enumerate(
            zip(all_onsets, next_onsets, conditions, errors)):
        if errortrials == 0 and err:
            continue
        elif errortrials == 1 and not err:
            continue

        start_fr = onset - pre_f
        if start_fr < 0:
            trial_pre_s = pre_s - (-start_fr * fr)
            start_fr = 0
        else:
            trial_pre_s = pre_s

        trial_result = []
        for replay_type in replay_types:
            trial_replay_result = classifier_results[replay_type][
                start_fr:next_onset - 1]
            time = np.arange(len(trial_replay_result)) / fr - trial_pre_s

            index = pd.MultiIndex.from_product(
                [[trial_idx], [cond], time],
                names=['trial_idx', 'condition', 'time'])
            trial_result.append(
                pd.Series(trial_replay_result, index=index, name=replay_type))
        result.append(pd.concat(trial_result, axis=1))

    final_result = pd.concat(result, axis=0)

    if len(final_result) and post_s is not None:
        final_result = final_result.loc(axis=0)[:, :, :post_s]

    return final_result


def _events_df(run, threshold=0.1, xmask=False):
    import pool
    t2p = run.trace2p()
    c2p = run.classify2p()
    events_list = [pd.DataFrame()]
    for event_type in pool.config.stimuli():
        events = c2p.events(
            event_type, threshold=threshold, traces=t2p, xmask=xmask)
        index = pd.MultiIndex.from_product(
            [[run.mouse], [run.date], [run.run], [event_type],
             np.arange(len(events))],
            names=['mouse', 'date', 'run', 'event_type', 'event_idx'])
        events_list.append(pd.DataFrame({'frame': events}, index=index))

    return pd.concat(events_list, axis=0)


def _frames_df(run):
    t2p = run.trace2p()
    frame_period = 1. / t2p.framerate
    frames = np.arange(t2p.nframes)
    index = pd.MultiIndex.from_product(
        [[run.mouse], [run.date], [run.run] * len(frames)],
        names=['mouse', 'date', 'run'])
    frames_df = pd.DataFrame({'frame': frames, 'frame_period': frame_period},
                             index=index)
    return frames_df


def _events_by_trial(run, threshold, xmask=False):
    t2p = run.trace2p()

    all_onsets = t2p.csonsets()
    conditions = t2p.conditions()
    errors = t2p.errors(cs=None)

    next_onsets = np.concatenate([all_onsets[1:], t2p.nframes], axis=None)
    prev_onsets = np.concatenate([0, all_onsets[:-1]], axis=None)

    fr = t2p.framerate

    events = _events_df(run, threshold, xmask=xmask)
    frames = _frames_df(run)

    result, times = [pd.DataFrame()], [pd.DataFrame()]
    for trial_idx, (onset, next_onset, prev_onset, cond, err) in enumerate(zip(
            all_onsets, next_onsets, prev_onsets, conditions, errors)):

        trial_events = events.loc[
            (events.frame >= prev_onset) & (events.frame < next_onset)].copy()
        trial_events -= onset
        trial_events['time'] = trial_events.frame / fr

        trial_frames = frames.loc[
            (frames.frame >= prev_onset) & (frames.frame < next_onset)].copy()
        trial_frames.frame -= onset
        trial_frames['time'] = trial_frames.frame * trial_frames.frame_period

        # add in trial_idx, condition, error
        trial_events = pd.concat(
            [trial_events], keys=[trial_idx], names=['trial_idx'])
        trial_events = pd.concat(
            [trial_events], keys=[cond], names=['condition'])
        trial_events = pd.concat(
            [trial_events], keys=[err], names=['error'])

        trial_frames = pd.concat(
            [trial_frames], keys=[trial_idx], names=['trial_idx'])

        result.append(trial_events)
        times.append(trial_frames)
    events_df = pd.concat(result)
    events_df = events_df.reorder_levels(
        ['mouse', 'date', 'run', 'trial_idx', 'condition', 'error',
         'event_type', 'event_idx'])
    events_df.drop(columns=['frame'], inplace=True)

    times_df = pd.concat(times)
    times_df = times_df.reorder_levels(
        ['mouse', 'date', 'run', 'trial_idx'])
    times_df.drop(columns=['frame'], inplace=True)

    return events_df, times_df


def _bin_events(events, times, edges, bin_labels):
    events['time_cat'] = pd.cut(
        events.time, edges, labels=bin_labels)
    times['time_cat'] = pd.cut(
        times.time, edges, labels=bin_labels)

    events_gb = (events
                 .groupby(
                     ['mouse', 'date', 'run', 'trial_idx', 'condition',
                      'error', 'event_type', 'time_cat'])
                 .count()
                 .dropna()
                 .rename(columns={'time': 'events'}))

    times_gb = (times
                .groupby(
                    ['mouse', 'date', 'run', 'trial_idx', 'time_cat',
                     'frame_period'])
                .count()
                .dropna()
                .reset_index('frame_period')
                .rename(columns={'time': 'frames'}))

    # There has to be a better way to do this (merge?)
    # Expand across event types so that the merge will add in empty values
    all_times = []
    for event_type in config.stimuli():
        all_times.append(times_gb
                         .assign(event_type=event_type)
                         .set_index('event_type', append=True))
    times_gb = pd.concat(all_times)

    result = pd.merge(
        events_gb.reset_index(['condition', 'error']),
        times_gb,
        how='right',
        on=['mouse', 'date', 'run', 'trial_idx', 'time_cat', 'event_type'])
    result = result.reset_index(['time_cat', 'event_type'])

    # Add in 0's
    def fill_values(df):
        df['condition'] = \
            df['condition'].fillna(method='ffill').fillna(method='bfill')
        df['error'] = df['error'].fillna(method='ffill').fillna(method='bfill')
        df['events'] = df['events'].fillna(0)
        return df

    result = (result
              .groupby(['mouse', 'date', 'run', 'trial_idx'])
              .apply(fill_values))

    # Reset error to a boolean
    result['error'] = result['error'].astype('bool')

    result['event_rate'] = \
        result.events / (result.frame_period * result.frames)

    return result
