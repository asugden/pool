"""Reactivation figure layouts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import IndexSlice as Idx
import seaborn as sns

from flow.misc.plotting import right_label

from .. import config
from ..plotting import reactivation as react
from .. import dataframes as dfs


def classifier_throughout_trials(runs, pre_s=-2, post_s=None, limit_conditions=False):
    """Layout reactivation probability trial plots.

    Lays out an array of 2*n_trial_types x n_replay_types array of plots
    and plots the classifier probability of each replay type through trials.

    Parameters
    ----------
    runs : RunSorter or list of Runs
    pre_s : float
        Time before stim to include in PSTH.
    post_s : float, optional
        Time after stim to include. If None, include all time up to next stim.

    Returns
    -------
    fig : matplotlib.pyplot.Figure

    """
    df = dfs.reactivation.trial_classifier_df(runs)

    if limit_conditions:
        trial_types = ['plus', 'neutral', 'minus']
    else:
        trial_types = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    replay_types = config.stimuli()

    fig, axs = plt.subplots(
        len(trial_types) * 2, len(replay_types), sharex=True, sharey=True,
        figsize=(9, 16))

    for axs_row, trial_type in zip(axs[::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.reactivation_probability_throughout_trials(
                ax, df, trial_type=trial_type, replay_type=replay_type,
                pre_s=pre_s, post_s=post_s, errortrials=0, label='correct')

    for axs_row, trial_type in zip(axs[1::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.reactivation_probability_throughout_trials(
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

    return fig


def event_distribution_throughout_trials(
        runs, pre_s=-5, post_s=10, exclude_window=(-0.1, 2.5), threshold=0.1,
        kind='kde', limit_conditions=False, inactivity_mask=False,
        **plot_kwargs):
    """
    Plot continuous event distribution throughout trial.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s, post_s : float
        Start and stop trial times to plot, relative to stim onset.
    exclude_window : 2-element tuple
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

    """
    df = dfs.reactivation.trial_events_df(
        runs, threshold=threshold, xmask=False,
        inactivity_mask=inactivity_mask)

    # Make sure Index levels are in the correct order
    df = df.reorder_levels(
        ['mouse', 'date', 'run', 'trial_idx', 'condition', 'error',
         'event_type', 'event_idx'])

    # Limit to only {'plus', 'neutral', 'minus'}
    if limit_conditions:
        df = df.loc[Idx[:, :, :, :, ['plus', 'neutral', 'minus'], :, :, :], :]
        row_order = ['plus', 'neutral', 'minus']
    else:
        row_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']

    df = df.reset_index(['event_type', 'error', 'condition'])

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

    return g


def binned_event_distrbituions_throughout_trials(
        runs, pre_s=-5, iti_start_s=5, iti_end_s=10, threshold=0.1, kind='bar',
        limit_conditions=False, inactivity_mask=False, **plot_kwargs):
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
    inactvitiy_mask : bool
        If True, limit reactivations to periods of inactivity.
    **plot_kwargs
        Additional arguments are passed directly to the plotting functions.

    Returns
    -------
    seaborn.GridSpec

    """
    # Update some plot_kwargs
    if kind == 'bar' and 'ci' not in plot_kwargs:
        plot_kwargs['ci'] = 68  # Plot SEM as error bars

    # These should probably match pool.analyses.trialdf event paddings
    pre_stim_pad = -0.1
    post_stim_pad = 2.5
    edges = [pre_s, pre_stim_pad, 0, 2, post_stim_pad, iti_start_s, iti_end_s]
    bin_labels = ['pre', 'pre-buffer', 'stim', 'post-buffer', 'post', 'iti']

    events = dfs.reactivation.trial_events_df(
        runs, threshold=threshold, xmask=False,
        inactivity_mask=inactivity_mask)
    frames = dfs.imaging.trial_frames_df(
        runs, inactivity_mask=inactivity_mask)

    row_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    if limit_conditions:
        row_order = ['plus', 'neutral', 'minus']
        events = (events
                  .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                   'condition', 'error', 'event_type',
                                   'event_idx'])
                  .loc[Idx[:, :, :, :, ['plus', 'neutral', 'minus']], :]
                  )

        frames = (frames
                  .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                   'condition', 'error'])
                  .loc[Idx[:, :, :, :, ['plus', 'neutral', 'minus']], :]
                  )

    events_binned = (dfs.bin_events(events, frames, edges, bin_labels)
                     .reorder_levels(['mouse', 'date', 'run', 'event_type',
                                      'condition', 'error', 'time_cat'])
                     .loc[Idx[:, :, :, :, :, :, ['pre', 'post', 'iti']], :]
                     .reset_index(['event_type', 'condition', 'error',
                                   'time_cat'])
                     )
    events_binned.time_cat.cat.remove_unused_categories(inplace=True)

    g = sns.catplot(
        x='time_cat', y='event_rate', col='event_type', row='condition',
        hue='error', data=events_binned, kind=kind, margin_titles=True,
        row_order=row_order, col_order=['plus', 'neutral', 'minus'],
        **plot_kwargs)

    g.set_xlabels('')
    g.set_ylabels('Event rate (Hz)')

    return g


def histogram_event_distrbituions_throughout_trials(
        runs, pre_s=-5, post_s=10, bin_size_s=1, exclude_window=(-0.1, 2.5), threshold=0.1, kind='bar',
        limit_conditions=False, inactivity_mask=False, **plot_kwargs):
    """
    Plot reactivation rates like a histogram (1-s bins) throughout trial.

    Parameters
    ----------
    runs : RunSorter or list of Run
    pre_s, post_s : float
        Start and stop trial times to plot, relative to stim onset.
    bin_size_s : float
        Bin size, in seconds.
    exclude_window : 2-element tuple
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

    """
    # Update some plot_kwargs
    if kind == 'bar' and 'ci' not in plot_kwargs:
        plot_kwargs['ci'] = 68  # Plot SEM as error bars

    left_edges = np.arange(pre_s, post_s, bin_size_s)
    edges = np.concatenate([left_edges, [left_edges[-1] + bin_size_s]])
    bin_labels = [str(x) for x in left_edges]

    events = dfs.reactivation.trial_events_df(
        runs, threshold=threshold, xmask=False,
        inactivity_mask=inactivity_mask)
    frames = dfs.imaging.trial_frames_df(
        runs, inactivity_mask=inactivity_mask)

    if exclude_window is not None:
        events = events.reset_index(['condition'])
        events = events[(events.condition == 'blank') |
                        ((events.time < exclude_window[0]) |
                        (events.time > exclude_window[1]))]
        events = (events
                  .set_index(['condition'], append=True)
                  .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                   'condition', 'error', 'event_type',
                                   'event_idx'])
                  )

    row_order = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    if limit_conditions:
        row_order = ['plus', 'neutral', 'minus']
        events = (events
                  .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                   'condition', 'error', 'event_type',
                                   'event_idx'])
                  .loc[Idx[:, :, :, :, ['plus', 'neutral', 'minus']], :]
                  )

        frames = (frames
                  .reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                                   'condition', 'error'])
                  .loc[Idx[:, :, :, :, ['plus', 'neutral', 'minus']], :]
                  )

    events_binned = (dfs.bin_events(events, frames, edges, bin_labels)
                     .reset_index(['event_type', 'condition', 'error',
                                   'time_cat'])
                     )

    # Not really sure why this is sometimes a categorical series and sometimes
    # not.
    # events_binned.time_cat.cat.remove_unused_categories(inplace=True)

    g = sns.catplot(
        x='time_cat', y='event_rate', col='event_type', row='condition',
        hue='error', data=events_binned, kind=kind, margin_titles=True,
        row_order=row_order, col_order=['plus', 'neutral', 'minus'],
        **plot_kwargs)

    g.set_xlabels('')
    g.set_ylabels('Event rate (Hz)')

    return g

# def binned_event_distrbituions_throughout_trials_orig(
#         runs, pre_s=-5, iti_start_s=5, iti_end_s=10, stim_pad_s=0.1,
#         threshold=0.1, kind='bar', limit_conditions=False, **plot_kwargs):
#     edges = [pre_s, -stim_pad_s, 0, 2, 2 + stim_pad_s, iti_start_s, iti_end_s]
#     bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']

#     all_events = []
#     for run in runs:
#         events, times = react._events_by_trial(run, threshold, xmask=False)

#         events_binned = react._bin_events(events, times, edges, bin_labels)
#         all_events.append(events_binned)
#     events_binned = pd.concat(all_events, axis=0)

#     events_binned = events_binned[
#         events_binned.time_cat.isin(['pre', 'post', 'iti'])]
#     events_binned.time_cat.cat.remove_unused_categories(inplace=True)

#     g = sns.catplot(
#         x='time_cat', y='event_rate', col='event_type', row='condition',
#         hue='error', data=events_binned, kind=kind, margin_titles=True,
#         row_order=['plus', 'neutral', 'minus', 'pavlovian', 'blank'],
#         col_order=['plus', 'neutral', 'minus'], **plot_kwargs)

#     g.set_xlabels('')
#     g.set_ylabels('Event rate (Hz)')

#     return g


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
