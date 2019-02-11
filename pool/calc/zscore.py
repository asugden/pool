"""Analyses directly related to zscoring."""
import numpy as np
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize


@memoize(across='date', updated=190130, returns='cell array')
def iti_mu(date, window=4):
    """
    Calculate the mean (mu) of daily trace per cell avoiding visual
    stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds before the visual stim to include in calcualtion.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are mean
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
        traces.append(t2p.d['dff'])

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # calculate masked mean per cell
    mu = np.nanmean(traces[:, masks], axis=1)

    return mu


@memoize(across='date', updated=190130, returns='cell array')
def iti_sigma(date, window=4):
    """
    Calculate the standard deviation (sigma) of daily trace per
    cell avoiding visual stimulus period.

    Parameters
    ----------
    date : Date
    window : int
        Number of seconds before the visual stim to include in calcualtion.

    Result
    ------
    np.ndarray
        An array of length equal to the number cells, values are standard
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
        traces.append(t2p.d['dff'])

    # concatenate across days
    traces = np.concatenate(traces, axis=1)
    masks = np.concatenate(masks, axis=0)

    # calculate masked standard deviation per cell
    sigma = np.nanstd(traces[:, masks], axis=1)

    return sigma
