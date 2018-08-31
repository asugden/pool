"""Stimulus plotting functions."""
import numpy as np
try:
    from bottleneck import nanmean, nanmedian
except ImportError:
    from numpy import nanmean, nanmedian

import pool


def stimulus_mean_response(
        ax, date, plot_all=False, start_s=-1, end_s=2, trace_type='dff',
        **kwargs):
    """Plot the mean stimulus response to each stim type.

    Parameters
    ----------
    ax : mpl.axes
    date : Date
    plot_all : bool
        If True, plot each cell individually instead of averaging.
    start_s : int
        Time before stim to include, in seconds.
    end_s : int
        Time after stim to include, in seconds.
    trace_type : {'dff', 'raw', 'deconvolved'}
        Type of trace to plot.
    **kwargs
        Additional keyword arguments are passed to t2p.cstraces().

    """
    adb = pool.database.db()
    colors = pool.config.colors()
    cses = ['plus', 'minus', 'neutral']
    framerate = None

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
    runs = date.runs(runtypes=['train', 'spontaneous'])
    framerate = runs[0].t2p.framerate
    for run in runs:
        t2p = run.t2p
        assert t2p.framerate == framerate
        for cs in cses:
            traces = t2p.cstraces(
                cs, start_s=start_s, end_s=end_s, trace_type=trace_type)
            if cs not in responses:
                responses[cs] = traces
            else:
                responses[cs] = np.concatenate([responses[cs], traces], 2)
    sort_order = np.array(adb.get('sort-simple', date.mouse, date.date))
    sort_borders = adb.get('sort-simple-borders', date.mouse, date.date)

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
    # from pudb import set_trace; set_trace()
