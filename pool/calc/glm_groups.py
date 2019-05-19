"""Analyses directly related to what cells are driven by."""
import numpy as np
from scipy import stats
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean

from ..database import memoize
from .. import stimulus
import flow.glm
try:
    from ..calc_legacy import driven
except ImportError:
    pass

@memoize(across='date', updated=190308, returns='value')
def responses(date,
              group,
              trange=(0, 1),
              driven_cs=None,
              driven_threshold=-1,
              positive_only=True,
              glm_type='simpglm'):
    """
    Return the mean coefficients for a GLM group.

    Parameters
    ----------
    date : Date object
    group : str
        GLM group, such as 'ensure', 'quinine', 'plus', 'lick'
    trange : tuple of floats
        The time range to integrate
    driven_cs : str
        If set to true, limit to visually driven cells
    driven_threshold : float
        The threshold of visual drivenness in inverse log p
    positive_only : bool
        Include only positive-ward values in the mean coefficient.
    glm_type : str {'simpglm', 'simpglm2', 'safeglm'}
        The GLM type to analyze

    Returns
    -------
    value
        The mean coefficients
    """

    glm = flow.glm.glm(date.mouse, date.date, glm_type=glm_type)
    unit = glm.protovector(group, trange=trange, err=-1)

    if positive_only:
        unit[unit < 0] = 0

    if driven_cs is not None and driven_threshold > -1:
        vdrive = driven.visually(date, driven_cs) > driven_threshold
        unit[np.invert(vdrive)] = 0

    return np.mean(unit).astype(np.float64)/len(unit)


@memoize(across='date', updated=190307, returns='value')
def fraction(date,
             group,
             driven_cs=None,
             driven_threshold=-1,
             exclude_lick_cells=False,
             exclusive_group=False,
             multiplexed_group=False,
             glm_type='simpglm',
             minimum_predictive_value=0.01,
             minimum_fraction=0.05):
    """
    Return the fraction of cells for a GLM group.

    Parameters
    ----------
    date : Date object
    group : str
        GLM group, such as 'ensure', 'quinine', 'plus', 'lick'
    driven_cs : str
        If set to a cs, plus, neutral, or minus, limit to visually driven cells
    driven_threshold : float
        The threshold of visual drivenness in inverse log p
    exclude_lick_cells : bool
        If True, remove any cells that respond to licking
    exclusive_group : bool
        If True, only include those cells that respond uniquley to the group
    multiplexed_group : bool
        Opposite of exclusive_group. If true, include only those
        cells that respond to both the group and other groups.
    glm_type : str {'simpglm', 'simpglm2', 'safeglm'}
        The GLM type to analyze
    minimum_predictive_value : float [0, 1]
        The minimum variance predicted by all glm filters
    minimum_fraction : float [0, 1]
        The minimum fraction of glm variance explained by filters for each cell type

    Returns
    -------
    value
        The fraction per cell
    """

    lbls = flow.glm.labels(date.mouse, date.date, minimum_predictive_value,
                           minimum_fraction, glm_type)

    groupname = group
    if exclusive_group:
        groupname += '-only'
    elif multiplexed_group:
        groupname += '-multiplexed'

    out = lbls[groupname]

    if driven_cs is not None and driven_threshold > -1:
        vdrive = driven.visually(date, driven_cs) > driven_threshold
        out[np.invert(vdrive)] = False

    if exclude_lick_cells:
        out[lbls['lick']] = False

    return np.sum(out).astype(np.float64)/len(out)


@memoize(across='date', updated=190307, returns='cell array')
def responsive(date,
               group,
               exclusive_group=False,
               multiplexed_group=False,
               glm_type='simpglm',
               minimum_predictive_value=0.01,
               minimum_fraction=0.05):
    """
    Return the fraction of cells for a GLM group.

    Parameters
    ----------
    date : Date object
    group : str
        GLM group, such as 'ensure', 'quinine', 'plus', 'lick'
    exclusive_group : bool
        If True, only include those cells that respond uniquley to the group
    multiplexed_group : bool
        Opposite of exclusive_group. If true, include only those
        cells that respond to both the group and other groups.
    glm_type : str {'simpglm', 'simpglm2', 'safeglm'}
        The GLM type to analyze
    minimum_predictive_value : float [0, 1]
        The minimum variance predicted by all glm filters
    minimum_fraction : float [0, 1]
        The minimum fraction of glm variance explained by filters for each cell type

    Returns
    -------
    cell array
        The fraction per cell
    """

    lbls = flow.glm.labels(date.mouse, date.date, minimum_predictive_value,
                           minimum_fraction, glm_type)

    groupname = group
    if exclusive_group:
        groupname += '-only'
    elif multiplexed_group:
        groupname += '-multiplexed'

    return lbls[groupname]
