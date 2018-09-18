"""Figure layouts for analyzing stimulus repsonses."""

import pool
from pool.plotting import stimulus as pps

def all_stimulus_traces(
        fig, date, roi_idx, start_s=-1, end_s=2, trace_type='dff', **kwargs):
    """Plots all stimuli responses for a single ROI."""
    cses = pool.config.stimuli()

    axs = [fig.add_subplot(1, len(cses), i + 1) for i in range(len(cses))]
    for ax, cs in zip(axs, cses):
        pps.all_stimulus_traces(
            ax, date, roi_idx, cs, start_s=start_s, end_s=end_s,
            trace_type=trace_type, **kwargs)
        if ax != axs[0]:
            ax.set_ylabel('')
    return fig
