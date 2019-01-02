"""Reactivation plotting functions."""
from pandas import IndexSlice as Idx

from .. import config


def trial_classifier_probability(
        ax, df, trial_type='plus', replay_type='plus',
        pre_s=-2, post_s=None, errortrials=-1, **kwargs):
    """Plot the classifier probability throughout the trial.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes
    runs : RunSorter
    trial_type : str
        Type of trial to plot data from.
    replay_type : str
        Type of replay to plot classifier probabilities for.
    pre_s, post_s : float, optional
        Time in seconds relative to stim onset to include. Negative value for
        before stim.
    errortrials : {-1, 0, 1}
        -1 is all trials, 0 is correct trials, 1 is error trials

    """
    # Add in a few more things to kwargs
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = config.colors(replay_type)
    if 'label' not in kwargs:
        kwargs['label'] = replay_type

    if errortrials == -1:
        error_slice = slice(None)
    elif errortrials == 0:
        error_slice = False
    elif errortrials == 1:
        error_slice = True

    # Always make sure levels are in expected order before slicing
    df = df.reorder_levels(['mouse', 'date', 'run', 'trial_idx',
                            'condition', 'error', 'time'])
    trimmed_df = df.loc[
        Idx[:, :, :, :, trial_type, error_slice, :],  # Row slice
        replay_type]                                  # Column slice

    grouped = trimmed_df.groupby(level=['time']).mean()

    # # We could have no trials of the desired error type (or trial type in general)
    # if len(grouped):
    # Uses the slice object to correctly handle 'None' as an argument
    to_plot = grouped.loc[slice(pre_s, post_s)]
    ax.plot(to_plot, **kwargs)
