"""Reactivation figure layouts."""
import matplotlib.pyplot as plt

from flow.misc.plotting import right_label

from .. import config
from ..plotting import reactivation as react


def reactivation_probability_throughout_trials(runs, pre_s=2, post_s=None):
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
