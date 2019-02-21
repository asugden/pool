"""Measures of population distance."""
import numpy as np
from scipy.spatial.distance import cosine
try:
    from bottleneck import nanmean, nanmedian
except ImportError:
    from numpy import nanmean, nanmedian

from ..database import memoize
from .. import stimulus
from . import cell_activity


@memoize(across='run', updated=190220, returns='cells by other')
def cosdist_continuous(
        run, group, tracetype='deconvolved', trange=(0, 1), rectify=False,
        exclude_outliers=False, remove_group=None):
    """
    Calculate the cosine distance between a GLM protovector and population.

    Parameters
    ----------
    run : Run
    group : str
        Group in GLM.groups.
    tracetype : str
        Type of trace to compare to the GLM protovector.
    trange : 2-element tuple of float
        Time range to look at for the protovector.
    rectify : bool
        If True, set negative values of the protovector to 0.
    exclude_outliers : bool
        Not currently supported, eventually exclude some cells from the
        calculation.
    remove_group : str, optional
        If not None, remove this group protovector from the result.

    Returns
    -------
    np.ndarray
        Array of length number of time points corresponding to the cosine
        distance between the specified protovector and the population response.

    """
    if exclude_outliers:
        raise NotImplementedError

    glm = run.parent.glm()
    unit = glm.protovector(
        group, trange=trange, rectify=rectify, err=-1,
        remove_group=remove_group)

    trace = run.trace2p().trace(tracetype)

    # scipy.spatial.distance.cdist should be able to do this, but as of 190207
    # it keeps silently crashing the kernel (at least in Jupyter notebooks)
    result = [cosine(trace_t, unit) for trace_t in trace.T]

    return np.array(result)


@memoize(across='date', updated=190220, returns='value')
def cosine_similarity_stimuli(
        date, cs, group, trace_type='deconvolved', start_s=0, end_s=1, trange_glm=(0, 1), rectify=True,
        exclude_outliers=True, remove_group=None):
    """
    Calculate the cosine distance between a GLM protovector and population.

    Parameters
    ----------
    date : Date
    group : str
        Group in GLM.groups.
    tracetype : str
        Type of trace to compare to the GLM protovector.
    trange : 2-element tuple of float
        Time range to look at for the protovector.
    rectify : bool
        If True, set negative values of the protovector to 0.
    exclude_outliers : bool
        Not currently supported, eventually exclude some cells from the
        calculation.
    remove_group : str, optional
        If not None, remove this group protovector from the result.

    Returns
    -------
    np.ndarray
        Array of length number of time points corresponding to the cosine
        distance between the specified protovector and the population response.

    """

    stimuli = stimulus.trials(date, cs, start_s=start_s, end_s=end_s, trace_type=trace_type)
    stimuli = nanmean(stimuli, axis=2)
    stimuli = nanmean(stimuli, axis=1)

    if exclude_outliers:
        keep = cell_activity.keep(date)
        stimuli = stimuli[keep]
    else:
        keep = np.zeros(len(stimuli)) < 1

    glm = date.glm()
    unit = glm.protovector(
        group, trange=trange_glm, rectify=rectify, err=-1,
        remove_group=remove_group)[keep]

    return 1.0 - cosine(stimuli, unit)
