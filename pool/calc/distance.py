"""Measures of population distance."""
import numpy as np
from scipy.spatial.distance import cosine

from ..database import memoize


@memoize(across='run', updated=190208)
def cosdist(
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

    result = [cosine(trace_t, unit) for trace_t in trace.T]

    return np.array(result)
