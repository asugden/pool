"""Reactivation figure layouts."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from flow.misc.plotting import right_label

from .. import config
from ..plotting import reactivation as react


def probability_throughout_trials(runs, pre_s=2, post_s=None):
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
    trial_types = ['plus', 'neutral', 'minus', 'pavlovian', 'blank']
    replay_types = config.stimuli()

    fig, axs = plt.subplots(
        len(trial_types) * 2, len(replay_types), sharex=True, sharey=True,
        figsize=(9, 16))

    for axs_row, trial_type in zip(axs[::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.reactivation_probability_throughout_trials(
                ax, runs, trial_type=trial_type, replay_type=replay_type,
                pre_s=pre_s, post_s=post_s, errortrials=0, label='correct')

    for axs_row, trial_type in zip(axs[1::2], trial_types):
        for ax, replay_type in zip(axs_row, replay_types):
            react.reactivation_probability_throughout_trials(
                ax, runs, trial_type=trial_type, replay_type=replay_type,
                pre_s=pre_s, post_s=post_s, errortrials=1, label='error',
                linestyle='--')

    for ax, replay_type in zip(axs[0, :], replay_types):
        ax.set_title(replay_type)
    for ax, trial_type in zip(axs[::2, -1], trial_types):
        right_label(ax, trial_type)
    for ax in axs[::2, 0]:
        ax.set_ylabel('correct\nreplay probability')
    for ax in axs[1::2, 0]:
        ax.set_ylabel('error\n')
    for ax in axs[-1, :]:
        ax.set_xlabel('Time from stim (s)')

    return fig


def event_distributions_throughout_trials(
        runs, pre_s=5, post_s=10, exclude_window=(-0.1, 2.1), threshold=0.1):
    all_events = []
    for run in runs:
        events, _ = react._events_by_trial(run, threshold)
        all_events.append(events)
    events = pd.concat(all_events, axis=0)

    events.reset_index(['event_type', 'error', 'condition'], inplace=True)

    events = events[
        (events.time > -pre_s) &
        (events.time < post_s) &
        ((events.condition == 'blank') |
            ((events.time < exclude_window[0]) |
                (events.time > exclude_window[1])))]

    g = sns.FacetGrid(
        events, col='event_type', row='condition', hue='error',
        row_order=['plus', 'neutral', 'minus', 'pavlovian', 'blank'],
        col_order=['plus', 'neutral', 'minus'], margin_titles=True)

    g.map(plt.axvline, x=0, ls=':', color='k')
    g.map(sns.distplot, 'time', rug=True, hist=False, kde=True)
    if exclude_window is not None:
        g.map(
            plt.fill_betweenx, y=plt.ylim(), x1=exclude_window[0],
            x2=exclude_window[1], color='0.9')

    g.set(xlim=(-pre_s, post_s), ylim=(0, 0.2))
    g.set_ylabels('Normalized density')
    g.add_legend()

    return g

def binned_event_distrbituions_throughout_trials(
        runs, pre_s=5, iti_start_s=5, iti_end_s=10, stim_pad_s=0.1,
        threshold=0.1, kind='bar'):
    edges = [-pre_s, -stim_pad_s, 0, 2, 2 + stim_pad_s, iti_start_s, iti_end_s]
    bin_labels = ['pre', 'pre_buffer', 'stim', 'post_buffer', 'post', 'iti']

    all_events = []
    for run in runs:
        events, times = react._events_by_trial(run, threshold, xmask=False)

        events_binned = react._bin_events(events, times, edges, bin_labels)
        all_events.append(events_binned)
    events_binned = pd.concat(all_events, axis=0)

    events_binned = events_binned[
        events_binned.time_cat.isin(['pre', 'post', 'iti'])]
    events_binned.time_cat.cat.remove_unused_categories(inplace=True)

    g = sns.catplot(
        x='time_cat', y='event_rate', col='event_type', row='condition',
        hue='error', data=events_binned, kind=kind, margin_titles=True, 
        row_order=['plus', 'neutral', 'minus', 'pavlovian', 'blank'],
        col_order=['plus', 'neutral', 'minus'])

    g.set_xlabels('')
    g.set_ylabels('Event rate (Hz)')

    return g
