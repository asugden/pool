"""Reactivation figure layouts."""
from builtins import str, zip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flow.misc.plotting import right_label

from .. import config
from ..plotting import reactivation as react
from .. import dataframes as dfs


def trial_classifier_probability(
        runs, pre_s=-5, post_s=None, exclude_window=(-0.2, 2.3),
        limit_conditions=False):
    """
    Layout reactivation probability trial plots.

    Lays out an array of 2*n_trial_types x n_replay_types array of plots
    and plots the classifier probability of each replay type through trials.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    pre_s : float
        Time before stim to include in PSTH.
    post_s : float, optional
        Time after stim to include. If None, include all time up to next stim.
    exclude_window : 2-element tuple, optional
        Start and stop times to exclude from plotting. Intended to mask out
        stimulus time.
    limit_conditions : bool
        If True, only look at plus, neutral, and minus conditions.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
    df : pd.DataFrame

    """
    classifier_df = dfs.reactivation.trial_classifier_df(runs)
    behav_df = dfs.behavior.behavior_df(runs)
    df = (dfs
          .smart_merge(classifier_df, behav_df, how='left', sort=False)
          .set_index(['condition', 'error'], append=True)
          .reorder_levels(['mouse', 'date', 'run', 'trial_idx', 'condition',
                           'error', 'time'])
          )

    if limit_conditions:
        trial_types = ['plus', 'neutral', 'minus']
    else:
        trial_types = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    replay_types = config.stimuli()

    if exclude_window is not None:
        df = df.reset_index(['time', 'condition'])
        mask = (df.condition != 'blank') & \
            (df.time >= exclude_window[0]) & \
            (df.time <= exclude_window[1])
        df.loc[mask, list(replay_types)] = np.nan
        df = (df
              .set_index(['time', 'condition'], append=True)
              .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                               'condition', 'error', 'time'])
              )

    fig, axs = plt.subplots(
        len(trial_types) * 2, len(replay_types), sharex=True, sharey=True,
        figsize=(9, 16))

    for axs_row, trial_type in zip(axs[::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.trial_classifier_probability(
                ax, df, trial_type=trial_type, replay_type=replay_type,
                pre_s=pre_s, post_s=post_s, errortrials=0, label='correct')

    for axs_row, trial_type in zip(axs[1::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.trial_classifier_probability(
                ax, df, trial_type=trial_type, replay_type=replay_type,
                pre_s=pre_s, post_s=post_s, errortrials=1, label='error',
                linestyle='--')

    for ax, replay_type in zip(axs[0, :], replay_types):
        ax.set_title('{} replays'.format(replay_type))
    for ax, trial_type in zip(axs[::2, -1], trial_types):
        right_label(ax, '{}\ntrials'.format(trial_type))
    for ax in axs[::2, 0]:
        ax.set_ylabel('correct\nreplay probability')
    for ax in axs[1::2, 0]:
        ax.set_ylabel('error\n')
    for ax in axs[-1, :]:
        ax.set_xlabel('Time from stim (s)')

    return fig, df


def trial_event_distributions(
        runs, pre_s=-5, post_s=10, exclude_window=(-0.2, 2.3), threshold=0.1,
        kind='kde', limit_conditions=False, inactivity_mask=False,
        **plot_kwargs):
    """
    Plot continuous event distribution throughout trial.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s, post_s : float
        Start and stop trial times to plot, relative to stim onset.
    exclude_window : 2-element tuple, optional
        Start and stop times to exclude from plotting. Intended to mask out
        stimulus time.
    threshold : float
        Classifier reactivation confidence threshold.
    kind : str
        Type of plot to plot.
    limit_conditions : bool
        If True, only look at plus, neutral, and minus conditions.
    inactvitiy_mask : bool
        If True, limit reactivations to periods of inactivity.
    **plot_kwargs
        Additional arguments are passed directly to the plotting functions.

    Returns
    -------
    seaborn.GridSpec
    pd.DataFrame

    """
    event_df = dfs.reactivation.trial_events_df(
        runs, threshold=threshold, xmask=False,
        inactivity_mask=inactivity_mask)
    behav_df = dfs.behavior.behavior_df(runs)
    df = dfs.smart_merge(event_df, behav_df, how='left', sort=True)

    # Limit to only {'plus', 'neutral', 'minus'}
    if limit_conditions:
        df = df.loc[(df.condition == 'plus') |
                    (df.condition == 'neutral') |
                    (df.condition == 'minus'), :]
        row_order = ['plus', 'neutral', 'minus']
    else:
        row_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']

    df = df[
        (df.time > pre_s) &
        (df.time < post_s) &
        ((df.condition == 'blank') |
            ((df.time < exclude_window[0]) |
                (df.time > exclude_window[1])))]

    g = sns.FacetGrid(
        df, col='event_type', row='condition', hue='error',
        row_order=row_order, col_order=['plus', 'neutral', 'minus'],
        margin_titles=True)

    g.map(plt.axvline, x=0, ls=':', color='k')
    if kind == 'kde':
        g.map(sns.distplot, 'time', rug=True, hist=False, kde=True,
              **plot_kwargs)
    else:
        raise ValueError("Unrecognized 'kind' argument.")
    g.add_legend()
    if exclude_window is not None:
        g.map(
            plt.fill_betweenx, y=plt.ylim(), x1=exclude_window[0],
            x2=exclude_window[1], color='0.9')

    g.set(xlim=(pre_s, post_s))
    g.set_xlabels('Time from stim onset (s)')
    g.set_ylabels('Normalized density')

    return g, df


def trial_event_labels(
        runs, pre_s=-5, iti_start_s=5, iti_end_s=10, threshold=0.1, kind='bar',
        limit_conditions=False, xmask=False, inactivity_mask=False,
        plot_bias=False, **plot_kwargs):
    """
    Plot reactivation rates in different intervals of the trial.

    By default plots pre-stimulus baseline, post-stimulus response window, and
    ITI period.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s : float
        Start of first interval, relative to stim onset.
    iti_start_s, iti_end_s : float
        Start and stop times (relative to stim onset) for the ITI period.
    threshold : float
        Classifier reactivation confidence threshold.
    kind : str
        Type of plot to plot.
    limit_conditions : bool
        If True, only look at plus, neutral, and minus conditions.
    xmask : bool
        If True, only allow one event (across types) per time bin.
    inactvitiy_mask : bool
        If True, limit reactivations to periods of inactivity.
    plot_bias : boolean
        If True, convert rates to relative rates across reactivation types.
    **plot_kwargs
        Additional arguments are passed directly to the plotting functions.

    Returns
    -------
    seaborn.GridSpec
    pd.DataFrame

    """
    # Update some plot_kwargs
    if kind == 'bar' and 'ci' not in plot_kwargs:
        plot_kwargs['ci'] = 68  # Plot SEM as error bars

    # These should probably match pool.analyses.trialdf event paddings
    pre_stim_pad = -0.2
    post_stim_pad = 2.6
    edges = [pre_s, pre_stim_pad, 0, 2, post_stim_pad, iti_start_s, iti_end_s]
    labels = ['pre', 'pre-buffer', 'stim', 'post-buffer', 'post', 'iti']

    # Get events and frames dfs merged with trial info
    behavior = dfs.behavior.behavior_df(runs)
    events = (dfs
              .reactivation.trial_events_df(
                  runs, threshold=threshold, xmask=xmask,
                  inactivity_mask=inactivity_mask)
              # Drop indexes which we don't want to merge bin on later
              .reset_index(['event_idx'], drop=True)
              .drop(columns=['trial_idx'])
              )
    events = dfs.smart_merge(events, behavior, how='left')
    frames = (dfs
              .imaging.trial_frames_df(
                  runs, inactivity_mask=inactivity_mask)
              .reset_index(['trial_idx', 'frame'], drop=True)
              )
    frames = dfs.smart_merge(frames, behavior, how='left')

    # Trim conditions if needed
    condition_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    if limit_conditions:
        condition_order = ['plus', 'neutral', 'minus']

        events = events.loc[
            events.condition.isin(['plus', 'minus', 'neutral']), :]
        frames = frames.loc[
            frames.condition.isin(['plus', 'minus', 'neutral']), :]

    # Do binning
    events_binned = dfs.bin_events(events, edges, labels)
    frames_binned = dfs.bin_events(frames, edges, labels)

    # Trim down to desired windows for both
    events_binned = events_binned.reset_index('bin')
    events_binned = (events_binned
                     .loc[events_binned['bin'].isin(
                         ['pre', 'post', 'iti']), :]
                     )
    events_binned.bin.cat.remove_unused_categories(inplace=True)
    events_binned = events_binned.set_index('bin', append=True)

    frames_binned = frames_binned.reset_index('bin')
    frames_binned = (frames_binned
                     .loc[frames_binned['bin'].isin(
                         ['pre', 'post', 'iti']), :]
                     )
    frames_binned.bin.cat.remove_unused_categories(inplace=True)
    frames_binned = frames_binned.set_index('bin', append=True)

    rate_df = dfs.event_rate(
        events_binned, frames_binned, event_label_col='event_type')

    if plot_bias:
        rate_df = rate_df.unstack('event_type')
        rate_sum = rate_df.sum(axis=1)
        rate_bias_df = (rate_df
                        .div(rate_sum, axis=0)
                        .dropna()
                        .stack()
                        .reset_index()
                        )

        g = sns.catplot(
            x='bin', y='event_rate', col='error', row='condition',
            hue='event_type', data=rate_bias_df, kind=kind,
            margin_titles=True, row_order=condition_order,
            palette=config.colors(), hue_order=config.stimuli(), **plot_kwargs)

        g.set_xlabels('')
        g.set_ylabels('Fraction of events')

        return g, rate_bias_df

    else:
        g = sns.catplot(
            x='bin', y='event_rate', col='event_type', row='condition',
            hue='error', data=rate_df.reset_index(), kind=kind,
            margin_titles=True, row_order=condition_order,
            col_order=['plus', 'neutral', 'minus'], **plot_kwargs)

        g.set_xlabels('')
        g.set_ylabels('Event rate (Hz)')

        return g, rate_df


def trial_event_bins(
        runs, pre_s=-5, post_s=10, bin_size_s=1, exclude_window=(-0.2, 2.3),
        threshold=0.1, kind='bar', limit_conditions=False,
        inactivity_mask=False, **plot_kwargs):
    """
    Plot reactivation rates like a histogram (1-s bins) throughout trial.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s, post_s : float
        Start and stop trial times to plot, relative to stim onset.
    bin_size_s : float
        Bin size, in seconds.
    exclude_window : 2-element tuple, optional
        Start and stop times to exclude from plotting. Intended to mask out
        stimulus time.
    threshold : float
        Classifier reactivation confidence threshold.
    kind : str
        Type of plot to plot.
    limit_conditions : bool
        If True, only look at plus, neutral, and minus conditions.
    inactvitiy_mask : bool
        If True, limit reactivations to periods of inactivity.
    **plot_kwargs
        Additional arguments are passed directly to the plotting functions.

    Returns
    -------
    seaborn.GridSpec
    pd.DataFrame

    """
    # Update some plot_kwargs
    if kind == 'bar' and 'ci' not in plot_kwargs:
        plot_kwargs['ci'] = 68  # Plot SEM as error bars

    left_edges = np.arange(pre_s, post_s, bin_size_s)
    edges = np.concatenate([left_edges, [left_edges[-1] + bin_size_s]])
    labels = [str(x) for x in left_edges]

    # Get events and frames dfs merged with trial info
    behavior = dfs.behavior.behavior_df(runs)
    events = (dfs
              .reactivation.trial_events_df(
                  runs, threshold=threshold, xmask=False,
                  inactivity_mask=inactivity_mask)
              # Drop indexes which we don't want to merge bin on later
              .reset_index(['event_idx'], drop=True)
              .drop(columns=['trial_idx'])
              )
    events = dfs.smart_merge(events, behavior, how='left')
    frames = (dfs
              .imaging.trial_frames_df(
                  runs, inactivity_mask=inactivity_mask)
              .reset_index(['trial_idx', 'frame'], drop=True)
              )
    frames = dfs.smart_merge(frames, behavior, how='left')

    # Cut out exclude window
    if exclude_window is not None:
        events = events[(events.condition == 'blank') |
                        ((events.time < exclude_window[0]) |
                        (events.time > exclude_window[1]))]

    # Trim conditions if needed
    condition_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    if limit_conditions:
        condition_order = ['plus', 'neutral', 'minus']

        events = events.loc[
            events.condition.isin(['plus', 'minus', 'neutral']), :]
        frames = frames.loc[
            frames.condition.isin(['plus', 'minus', 'neutral']), :]

    # Do binning
    events_binned = dfs.bin_events(events, edges, labels)
    frames_binned = dfs.bin_events(frames, edges, labels)

    rate_df = dfs.event_rate(
        events_binned, frames_binned, event_label_col='event_type')

    g = sns.catplot(
        x='bin', y='event_rate', col='event_type', row='condition',
        hue='error', data=rate_df.reset_index(), kind=kind, margin_titles=True,
        row_order=condition_order, col_order=['plus', 'neutral', 'minus'],
        **plot_kwargs)

    g.set_xlabels('')
    g.set_ylabels('Event rate (Hz)')

    return g, rate_df


def trigger_event_bins(
        runs, pre_s=-1, post_s=5, bin_size_s=0.5,
        threshold=0.1, kind='bar', inactivity_mask=False, **plot_kwargs):
    """
    Plot reactivation rates like a histogram (1-s bins) throughout trial.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s, post_s : float
        Start and stop trial times to plot, relative to stim onset.
    bin_size_s : float
        Bin size, in seconds.
    threshold : float
        Classifier reactivation confidence threshold.
    kind : str
        Type of plot to plot.
    inactvitiy_mask : bool
        If True, limit reactivations to periods of inactivity.
    **plot_kwargs
        Additional arguments are passed directly to the plotting functions.

    Returns
    -------
    seaborn.GridSpec
    pd.DataFrame

    """
    # Update some plot_kwargs
    if kind == 'bar' and 'ci' not in plot_kwargs:
        plot_kwargs['ci'] = 68  # Plot SEM as error bars

    left_edges = np.arange(pre_s, post_s, bin_size_s)
    edges = np.concatenate([left_edges, [left_edges[-1] + bin_size_s]])
    labels = [str(x) for x in left_edges]

    all_events, all_frames = [], []
    for trigger in ['reward', 'punishment', 'lickbout']:
        trig_events = (dfs.reactivation
                       .trigger_events_df(
                           runs, trigger, threshold=threshold, xmask=False,
                           inactivity_mask=inactivity_mask)
                       .reset_index(['event_idx'], drop=True)
                       .drop(columns=['trigger_idx'])
                       .pipe(dfs.bin_events, edges, labels)
                       .assign(trigger=trigger)
                       .set_index('trigger', append=True)
                       )
        trig_frames = (dfs.imaging
                       .trigger_frames_df(
                           runs, trigger, inactivity_mask=inactivity_mask)
                       .reset_index(['trigger_idx', 'frame'], drop=True)
                       .pipe(dfs.bin_events, edges, labels)
                       .assign(trigger=trigger)
                       .set_index('trigger', append=True)
                       )

        all_events.append(trig_events)
        all_frames.append(trig_frames)
    events = pd.concat(all_events, axis=0)
    frames = pd.concat(all_frames, axis=0)

    rate_df = dfs.event_rate(events, frames, event_label_col='event_type')

    g = sns.catplot(
        x='bin', y='event_rate', col='event_type', row='trigger',
        data=rate_df.reset_index(), kind=kind, margin_titles=True,
        **plot_kwargs)

    g.set_xlabels('')
    g.set_ylabels('Event rate (Hz)')

    return g, rate_df


def peri_event_behavior(df, limit_conditions=False, **plot_kwargs):
    """Show mean before performance before and after each replay event.

    Parameters
    ----------
    df : pd.DataFrame
        pool.dataframes.reactivation.peri_event_behavior_df
    limit_conditions : bool
        If True, only include conditions that match possible events types.

    """
    grouped = (df
               .groupby(
                   ['mouse', 'date', 'condition', 'event_type', 'trial_idx'])
               .mean()
               .reset_index())

    if limit_conditions:
        grouped = grouped[
            grouped['condition'].isin(['plus', 'neutral', 'minus'])]
        row_order = ['plus', 'neutral', 'minus']
    else:
        row_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']

    g = sns.catplot(
        x='trial_idx', y='error', col='event_type', row='condition',
        data=grouped, kind='bar', margin_titles=True,
        row_order=row_order, col_order=['plus', 'neutral', 'minus'],
        **plot_kwargs)

    return g
