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

    all_results = []
    for run in runs:
        all_results.append(_classifier_by_trial(
            run.classify2p(), run.trace2p(), pre_s=pre_s, post_s=post_s,
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


def _classifier_by_trial(c2p, t2p, pre_s=2, post_s=None, errortrials=-1):
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

    result = []
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

    if post_s is not None:
        final_result = final_result.loc(axis=0)[:, :, :post_s]

    return final_result
