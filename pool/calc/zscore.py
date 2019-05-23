"""Analyses directly related to z-scoring."""
import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize


@memoize(across='date', updated=190519, returns='cell array')
def stim_min(date, window=5, nan_artifacts=False, thresh=20):
    """
    Calculate the min (mn) of daily trace per cell using only visual
    stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds after the visual onset to include in calculation.
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are mean
        cellular response during the pre-visual stim window.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces, masks = [], []
    for run in runs:

        t2p = run.trace2p()

        # create window for convolution
        win = np.concatenate((np.zeros(int(t2p.framerate*window)),
                             np.ones(int(t2p.framerate*window))))

        # create onsets vector and convolve onsets with window
        mask = np.zeros(t2p.nframes)
        mask[t2p.csonsets()] = 1
        mask = np.isin(np.convolve(mask, win, mode='same'), 1)

        masks.append(mask)
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked min per cell
    mn = np.nanmin(traces[:, masks], axis=1)

    return mn


@memoize(across='date', updated=190519, returns='cell array')
def stim_max(date, window=5, nan_artifacts=False, thresh=20):
    """
    Calculate the max (mx) of daily trace per cell using only visual
    stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds after the visual onset to exclude in calculation.
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are mean
        cellular response during the pre-visual stim window.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces, masks = [], []
    for run in runs:

        t2p = run.trace2p()

        # create window for convolution
        win = np.concatenate((np.zeros(int(t2p.framerate*window)),
                             np.ones(int(t2p.framerate*window))))

        # create onsets vector and convolve onsets with window
        mask = np.zeros(t2p.nframes)
        mask[t2p.csonsets()] = 1
        mask = np.isin(np.convolve(mask, win, mode='same'), 1)

        masks.append(mask)
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dialate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked max per cell
    mx = np.nanmax(traces[:, masks], axis=1)

    return mx


@memoize(across='date', updated=190212, returns='cell array')
def iti_mu(date, window=4, nan_artifacts=False, thresh=20):
    """
    Calculate the mean (mu) of daily trace per cell avoiding visual
    stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds before the visual stim to include in calculation.
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are mean
        cellular response during the pre-visual stim window.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces, masks = [], []
    for run in runs:

        t2p = run.trace2p()

        # create window for convolution
        win = np.concatenate((np.ones(int(t2p.framerate*window)),
                             np.zeros(int(t2p.framerate*window))))

        # create onsets vector and convolve onsets with window
        mask = np.zeros(t2p.nframes)
        mask[t2p.csonsets()] = 1
        mask = np.isin(np.convolve(mask, win, mode='same'), 1)

        masks.append(mask)
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked mean per cell
    mu = np.nanmean(traces[:, masks], axis=1)

    return mu


@memoize(across='date', updated=190212, returns='cell array')
def iti_sigma(date, window=4, nan_artifacts=False, thresh=20):
    """
    Calculate the standard deviation (sigma) of daily trace per
    cell avoiding visual stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds before the visual stim to include in calculation.
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are standard
        deviation of cellular response during the pre-visual-stim window.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces, masks = [], []
    for run in runs:

        t2p = run.trace2p()

        # create window for convolution
        win = np.concatenate((np.ones(int(t2p.framerate*window)),
                             np.zeros(int(t2p.framerate*window))))

        # create onsets vector and convolve onsets with window
        mask = np.zeros(t2p.nframes)
        mask[t2p.csonsets()] = 1
        mask = np.isin(np.convolve(mask, win, mode='same'), 1)

        masks.append(mask)
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked standard deviation per cell
    sigma = np.nanstd(traces[:, masks], axis=1)

    return sigma


@memoize(across='date', updated=190212, returns='cell array')
def mu(date, nan_artifacts=False, thresh=20):
    """
    Calculate the mean (mu) of daily trace per cell.

    Parameters
    ----------
    date : Date
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are mean
        cellular response.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces = []
    for run in runs:
        t2p = run.trace2p()
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked mean per cell
    mu = np.nanmean(traces, axis=1)

    return mu


@memoize(across='date', updated=190212, returns='cell array')
def sigma(date, nan_artifacts=False, thresh=20):
    """
    Calculate the standard deviation (sigma) of daily trace per
    cell.

    Parameters
    ----------
    date : Date
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are standard
        deviation of cellular response.

    """

    # get traces for whole day and a mask vector for pre-stim period
    runs = date.runs()
    traces = []
    for run in runs:
        t2p = run.trace2p()
        traces.append(t2p.trace('dff'))

    # concatenate across days
    traces = np.concatenate(traces, axis=1)

    # remove artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked standard deviation per cell
    sigma = np.nanstd(traces, axis=1)

    return sigma


@memoize(across='run', updated=190212, returns='cell array')
def run_mu(run, nan_artifacts=False, thresh=20):
    """
    Calculate the mean (mu) of single run trace per cell.

    Parameters
    ----------
    run : Run
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are mean
        cellular response.

    """

    # get traces for whole day and a mask vector for pre-stim period
    t2p = run.trace2p()
    traces = t2p.trace('dff')

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked mean per cell
    mu = np.nanmean(traces, axis=1)

    return mu


@memoize(across='run', updated=190212, returns='cell array')
def run_sigma(run, nan_artifacts=False, thresh=20):
    """
    Calculate the standard deviation (sigma) of single run trace per
    cell.

    Parameters
    ----------
    date : Date
    nan_artifacts : bool
        Remove regions where dff values are crazy
    thresh : int
        Threshold in dff space for artifact removal (e.g. 5 = 500% dff)

    Result
    ------
    np.ndarray
        An array of length equal to the number cells. Values are standard
        deviation of cellular response.

    """

    # get traces for single run and a mask vector for pre-stim period
    t2p = run.trace2p()
    traces = t2p.trace('dff')

    # remove huge artifacts from trace: blank with nans
    if nan_artifacts:
        noise = np.zeros(np.shape(traces))
        noise[np.abs(traces) > thresh] = 1

        # dilate blanking around threshold crossings
        for cell in range(np.shape(traces)[0]):
            noise[cell, :] = np.convolve(noise[cell, :], np.ones(3), mode='same')
        traces[noise != 0] = np.nan

    # calculate masked standard deviation per cell
    sigma = np.nanstd(traces, axis=1)

    return sigma
