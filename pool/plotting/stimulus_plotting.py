"""Stimulus plotting functions."""
import numpy as np

import pool

def stimulus_mean_response(ax, date, cses=None):
    """Plot the mean stimulus response to each stim type.

    Parameters
    ----------
    ax : mpl.axes
    date : Date
    cses : list

    """
    adb = pool.database.db()
    colors = pool.config.colors()
    args = {}
    if cses is None:
        cses = ['plus', 'minus', 'neutral']

    responses = {}
    for run in date.runs(runtypes=['train', 'spontaneous']):
        t2p = run.t2p
        for cs in cses:
            traces = t2p.cstraces(cs, args)
            if cs not in responses:
                responses[cs] = traces
            else:
                responses[cs] = np.concatenate([responses[cs], traces], 2)
    sort_order = adb.get('sort-order', date.mouse, date.date)
    sort_borders = adb.get('sort-borders', date.mouse, date.date)
    from pudb import set_trace; set_trace()
