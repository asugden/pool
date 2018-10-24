"""Figure layouts for analyzing stimulus responses."""
import matplotlib.pyplot as plt

import flow
import pool
from pool.plotting import stimulus as pps


def trial_traces(
        date, roi_idx, t_range_s=(-1, 2), trace_type='dff', cses=None,
        fig_kw=None, **kwargs):
    """Plots all stimuli responses for a single ROI.

    Parameters
    ----------
    date : Date
        Date object to analyze.
    roi_idx : int
        ROI index to analyze.
    t_range_s : tuple of int
        2 element tuple of start and end time relative to stimulus (in seconds).
    trace_type : {'dff', 'raw', 'deconvolved'}
        Type of trace to plot.
    cses : list of str, optional
        List of stimuli to plot. If None, defaults to cses in config file.
    fig_kw : dict
        Keyword arguments to be passed to the figure-generating
        function.
    **kwargs
        Additional keyword arguments are passed to the stim trace plotter.

    Returns
    -------
    fig : Figure

    """
    if cses is None:
        cses = pool.config.stimuli()
    if fig_kw is None:
        fig_kw ={}

    fig, axs = plt.subplots(1, len(cses), **fig_kw)
    for ax, cs in zip(axs, cses):
        pps.trial_traces(
            ax, date, roi_idx, cs, t_range_s=t_range_s, trace_type=trace_type,
            **kwargs)
        if ax != axs[0]:
            ax.set_ylabel('')
    return fig


def stimulus_response(
        dates, t_range_s, trace_type, sharey=False, **kwargs):
    """Layout and plot mean response to each stimulus across days.

    Parameters
    ----------
    dates : DateSorter
        DateSorter object to iterate.
    t_range_s : tuple of int
        2 element tuple of start and end time relative to stimulus (in seconds).
    trace_type : {'dff', 'raw', 'deconvolved'}
    sharey : bool
        If True, match all y scales.
    **kwargs
        Additional keyword arguments are passed to the actual plotting function.

    Returns
    -------
    fig : Figure

    """
    fig, axs = flow.misc.plotting.layout_subplots(
        len(dates), width=16, height=9, sharey=sharey, sharex=True)
    for date, ax in zip(dates, axs.flat):
        pps.stimulus_mean_response(
            ax, date, plot_all=False, trace_type=trace_type,
            t_range_s=t_range_s, **kwargs)

    axs.flat[0].legend(frameon=False)
    for ax in axs[:-1, :].flat:
        ax.set_xlabel('')
    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')

    return fig
