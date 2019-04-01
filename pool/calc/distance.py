"""Measures of population distance."""
import itertools as it
from multiprocessing import cpu_count, Pool
import numpy as np
from scipy.spatial.distance import cosine
try:
    from bottleneck import nanmean, nanmedian, nanmax
except ImportError:
    from numpy import nanmean, nanmedian, nanmax
import warnings

from ..database import memoize
from .. import stimulus
from . import cell_activity


@memoize(across='run', updated=190226, returns='other')
def cosine_similarity_continuous(
        run, group, tracetype='deconvolved', trange=(0, 1), rectify=False,
        exclude_outliers=False, remove_group=None, drop_glm_zeros=False):
    """
    Calculate the cosine distance between a GLM protovector and the population.

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
        If True, set negative values of the reconstructed vector (before
        averaging acoss `trange`) to 0.
    exclude_outliers : bool
        Not currently supported, eventually exclude some cells from the
        calculation.
    remove_group : str, optional
        If not None, remove this group protovector from the result.
    drop_glm_zeros : bool
        If True, remove all cells in which the GLM protovector is <= 0.

    Returns
    -------
    np.ndarray
        Array of length number of time points corresponding to the cosine
        similarity between the specified protovector and the population
        response. 1==most similar, 0==orthogonal, -1==anti-correlated

    """

    glm = run.parent.glm()
    unit = glm.protovector(
        group, trange=trange, rectify=rectify, err=-1,
        remove_group=remove_group)
    trace = run.trace2p().trace(tracetype)

    if exclude_outliers:
        keep = cell_activity.keep(run.parent, run_type=run.run_type)
        unit = unit[keep]
        trace = trace[keep, :]

    if drop_glm_zeros:
        keep = unit > 0
        unit = unit[keep]
        trace = trace[keep, :]

    n_processes = cpu_count() - 2
    pool = Pool(processes=n_processes)
    n_frames = trace.shape[1]
    chunksize = min(200, n_frames // n_processes)
    result = np.empty(n_frames, dtype=float)
    for idx, res in enumerate(pool.imap(
            _unpack_cosine, it.izip(trace.T, it.repeat(unit)),
            chunksize=chunksize)):
        result[idx] = res
    pool.close()

    # scipy.spatial.distance.cdist should be able to do this, but as of 190207
    # it keeps silently crashing the kernel (at least in Jupyter notebooks)
    # result = [cosine(trace_t, unit) for trace_t in trace.T]

    return 1.0 - np.array(result)


def _unpack_cosine(a):
    """Unpacker for parallelizing cosine calc."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cosine(a[0], a[1])
    return result


@memoize(across='date', updated=190226, returns='value')
def cosine_similarity_stimuli(
        date, cs, group, trace_type='deconvolved', start_s=0, end_s=1, trange_glm=(0, 1), rectify=False,
        exclude_outliers=False, remove_group=None, offset_glm_positive=False, remove_baseline_stimuli=False,
        drop_glm_zeros=False, max_across_trial=False, compare_to_trials=False, error_trials=0, binarize=None,
        correlation=False):
    """
    Calculate the cosine distance between a GLM protovector and population.

    Parameters
    ----------
    date : Date
    cs : str
        The stimulus type to compare to.
    group : str
        Group in GLM.groups, e.g. 'ensure'.
    trace_type : str
        Type of trace to compare to the GLM protovector.
    start_s : int
        Start time of stimulus
    end_s : int
        End time of stimulus
    trange_glm : 2-element tuple of float
        Time range to look at for the protovector.
    rectify : bool
        If True, set negative values of the protovector to 0.
    exclude_outliers : bool
        If True, exclude cells with the highest activity.
    remove_group : str, optional
        If not None, remove this group protovector from the result.
    offset_glm_positive : bool
        If True, subtract the minimum from the GLM protovector to force all values positive.
    remove_baseline_stimuli : bool
        Not yet implemented. If true, remove offsets, running, and brain motion from stimuli.
    drop_glm_zeros : bool
        If True, remove all cells in which the GLM protovector is <= 0.
    max_across_trial : bool
        If True, take the max across trials rather than the mean.
    compare_to_trials : bool
        If True, compare to individual trials rather than the mean trial.
    error_trials : int {-1, 0, 1}
        If -1, all trials, 0, correct trials, 1, incorrect trials.
    binarize : bool
        Not implemented yet. If true, binarize both vectors.
    correlation : bool
        If true, use correlation rather than cosine distance.

    Returns
    -------
    np.ndarray
        Array of length number of time points corresponding to the cosine
        distance between the specified protovector and the population response.

    """

    if remove_baseline_stimuli:
        raise NotImplementedError('Have not yet made it possible to remove brainmotion, running, and offsets from stimuli.')

    if binarize:
        raise NotImplementedError('Have not yet identified values for GLM and stim binarization.')

    stimuli = stimulus.trials(date, cs, start_s=start_s, end_s=end_s,
                              trace_type=trace_type, error_trials=error_trials)
    if max_across_trial:
        stimuli = nanmax(stimuli, axis=1)
    else:
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

    if offset_glm_positive and np.min(unit) < 0:
        unit -= np.min(unit)

    if drop_glm_zeros:
        keep = unit > 0
        unit = unit[keep]
        stimuli = stimuli[keep]

    if compare_to_trials:
        out = []
        if correlation:
            for s in range(np.shape(stimuli)[1]):
                out.append(np.corrcoef(stimuli[:, s], unit)[0, 1])
        else:
            for s in range(np.shape(stimuli)[1]):
                out.append(1.0 - cosine(stimuli[:, s], unit))
        return nanmean(out)
    else:
        stimuli = nanmean(stimuli, axis=1)
        if correlation:
            return np.corrcoef(stimuli, unit)[0, 1]
        else:
            return 1.0 - cosine(stimuli, unit)


# @memoize(across='run', updated=190220, returns='value')
def cosine_similarity_stimuli_run(
        run, cs, group, trace_type='deconvolved', start_s=0, end_s=1, trange_glm=(0, 1), rectify=True,
        exclude_outliers=True, remove_group=None, offset_glm_positive=False, remove_baseline_stimuli=False,
        drop_zeros=False, max_across=False, compare_to_trials=False):
    """
    Calculate the cosine distance between a GLM protovector and population.

    Parameters
    ----------
    run : Date
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

    stimuli = stimulus.trials(run, cs, start_s=start_s, end_s=end_s, trace_type=trace_type, error_trials=0)

    stimuli = nanmean(stimuli, axis=2)
    stimuli = nanmean(stimuli, axis=1)

    if exclude_outliers:
        keep = cell_activity.keep(run)
        stimuli = stimuli[keep]
    else:
        keep = np.zeros(len(stimuli)) < 1

    glm = run.parent.glm()
    unit = glm.protovector(
        group, trange=trange_glm, rectify=rectify, err=-1,
        remove_group=remove_group)[keep]

    if offset_glm_positive and np.min(unit) < 0:
        unit -= np.min(unit)

    if drop_zeros:
        keep = unit > 0
        unit = unit[keep]
        stimuli = stimuli[keep]

    return 1.0 - cosine(stimuli, unit)


@memoize(across='run', updated=190226, returns='other')
def correlation_continuous(
        run, group, tracetype='deconvolved', trange=(0, 1), rectify=False,
        exclude_outliers=False, remove_group=None, drop_glm_zeros=False):
    """
    Calculate the correlation between a GLM protovector and the population.

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
        If True, set negative values of the reconstructed vector (before
        averaging acoss `trange`) to 0.
    drop_glm_zeros : bool
        If True, remove all cells in which the GLM protovector is <= 0.
    exclude_outliers : bool
        Not currently supported, eventually exclude some cells from the
        calculation.
    remove_group : str, optional
        If not None, remove this group protovector from the result.

    Returns
    -------
    np.ndarray
        Array of length number of time points corresponding to the correlation
        between the specified protovector and the population response.

    """
    glm = run.parent.glm()
    unit = glm.protovector(
        group, trange=trange, rectify=rectify, err=-1,
        remove_group=remove_group)
    trace = run.trace2p().trace(tracetype)

    if exclude_outliers:
        keep = cell_activity.keep(run.parent, run_type=run.run_type)
        unit = unit[keep]
        trace = trace[keep, :]

    if drop_glm_zeros:
        keep = unit > 0
        unit = unit[keep]
        trace = trace[keep, :]

    n_processes = cpu_count() - 2
    pool = Pool(processes=n_processes)
    n_frames = trace.shape[1]
    chunksize = min(200, n_frames // n_processes)
    result = np.empty(n_frames, dtype=float)
    for idx, res in enumerate(pool.imap(
            np.corrcoef, it.izip(trace.T, it.repeat(unit)),
            chunksize=chunksize)):
        result[idx] = res[0, 1]
    pool.close()

    # Pool calculation should be equivalent to this:
    # result = [np.corrcoef(trace_t, unit)[0, 1] for trace_t in trace.T]

    return np.array(result)
