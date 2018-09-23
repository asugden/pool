"""Stimulus plotting functions."""
import numpy as np
try:
    from bottleneck import nanmean, nanmedian
except ImportError:
    from numpy import nanmean, nanmedian

import flow
import pool


def stimulus_mean_response(
        ax, date, plot_all=False, t_range_s=(-1, 2), trace_type='dff',
        cses=None, **kwargs):
    """Plot the mean stimulus response to each stim type.

    Classifies each ROI by peak response to one of the stims, or inhibited.
    Plots the mean of each category of cell.

    Parameters
    ----------
    ax : mpl.axes
    date : Date
    plot_all : bool
        If True, plot each cell individually instead of averaging.
    t_range_s : tuple of int
        2 element tuple of start and end time relative to stimulus (in seconds).
    trace_type : {'dff', 'raw', 'deconvolved'}
        Type of trace to plot.
    cses : list of str, optional
        List of stimuli to plot. If None, defaults to cses in config file.
    **kwargs
        Additional keyword arguments are passed to t2p.cstraces().

    """
    start_s, end_s = t_range_s
    if cses is None:
        cses = pool.config.stimuli()
    adb = pool.database.db()
    colors = pool.config.colors()

    # Use mean for spikes and median for fluorescence
    if trace_type == 'deconvolved':
        mean = nanmean
        ylabel = 'inferred spike count'
    else:
        mean = nanmedian
        if trace_type == 'dff':
            ylabel = 'dFF'
        else:
            ylabel = 'Raw fluorescence'

    responses = {}
    runs = date.runs(run_types=['training', 'spontaneous'])
    framerate = runs[0].trace2p().framerate
    for run in runs:
        t2p = run.trace2p()
        assert t2p.framerate == framerate
        for cs in cses:
            traces = t2p.cstraces(
                cs, start_s=start_s, end_s=end_s, trace_type=trace_type,
                **kwargs)
            if cs not in responses:
                responses[cs] = traces
            else:
                responses[cs] = np.concatenate([responses[cs], traces], 2)
    sort_order = np.array(adb.get('sort_order', date.mouse, date.date))
    sort_borders = adb.get('sort_borders', date.mouse, date.date)

    for cs in cses:
        start_idx = sort_borders[cs]
        borders = np.array(sort_borders.values())
        try:
            end_idx = borders[borders > start_idx].min()
        except ValueError:
            end_idx = len(sort_order)
        idxs = sort_order[start_idx:end_idx]

        color = colors.get(cs, 'k')
        x = np.arange(responses[cs][idxs].shape[1])/framerate + start_s

        if plot_all:
            roi_means = responses[cs][idxs].mean(2)
            for roi in roi_means:
                ax.plot(x, roi, c=color)
        else:
            data_mean = mean(mean(responses[cs][idxs], 2), 0)
            # median absolute deviation instead of std with median?
            data_std = mean(responses[cs][idxs], 2).std(0)

            ax.plot(x, data_mean, c=color, label=cs)
            ax.fill_between(
                x, data_mean - data_std, data_mean + data_std, color=color,
                alpha=0.2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    ax.set_title('{} - {}'.format(date.mouse, date.date))
    ax.set_xticks([start_s, 0, end_s])
    ax.legend()


def trial_traces(
        ax, date, roi_idx, stim_type, t_range_s=(-1, 2), trace_type='dff',
        normalize=False, **kwargs):
    """"Plot all trials as individual staggered traces."""
    start_s, end_s = t_range_s

    traces, errors = [], []
    runs = date.runs(run_types=['training'])
    framerate = runs[0].trace2p().framerate
    for run in runs:
        t2p = run.trace2p()
        assert t2p.framerate == framerate
        all_traces = t2p.cstraces(
            stim_type, start_s=start_s, end_s=end_s, trace_type=trace_type,
            **kwargs)
        traces.append(all_traces[roi_idx])
        errors.extend(t2p.errors(stim_type))
    traces = np.concatenate(traces, axis=1)

    flow.misc.plotting.plot_traces(
        ax, traces, (start_s, end_s), normalize=normalize, errors=errors)

    ax.set_title(stim_type)