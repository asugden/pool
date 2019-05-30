import numpy as np

import flow.categories
import flow.netcom
from ..database import memoize
from . import correlations
from . import driven

try:
    from ..calc_legacy import driven as driven_legacy
    from ..calc_legacy import correlations as correlations_legacy
except ImportError:
    pass


@memoize(across='date', updated=190312, returns='cell array')
def reward(date, visual_drivenness=50, correlation='noise', reward_cells_required=1):
    """
    Generate arrays determining whether cells are in reward clusters,
    or non-reward clusters.

    Parameters
    ----------
    date : Date instance
    visual_drivenness : int
        The amount to which a cell is visually driven
    correlation : str {'noise'}
        The correlation type used for clustering
    reward_cells_required : int
        The number of food-cue-reward cells in a cluster required to define
        it as a reward cluster.

    Returns
    -------
    boolean array
        True means cells are in reward clusters
        (Cells can be false in both reward and nonreward arrays)
    """

    non, rew = _reward_non(date, visual_drivenness=50,
                           correlation='noise', reward_cells_required=1)
    return rew


@memoize(across='date', updated=190312, returns='cell array')
def nonreward(date, visual_drivenness=50, correlation='noise', reward_cells_required=1):
    """
    Generate arrays determining whether cells are in reward clusters,
    or non-reward clusters.

    Parameters
    ----------
    date : Date instance
    visual_drivenness : int
        The amount to which a cell is visually driven
    correlation : str {'noise'}
        The correlation type used for clustering
    reward_cells_required : int
        The number of food-cue-reward cells in a cluster required to define
        it as a reward cluster.

    Returns
    -------
    tuple of boolean arrays
        True means cells are in non-reward clusters,
        (Cells can be false in both reward and nonreward arrays)
    """

    non, rew = _reward_non(date, visual_drivenness=50,
                           correlation='noise', reward_cells_required=1)
    return non


@memoize(across='date', updated=190301, returns='cell array')
def number_legacy(date, cs, visual_drivenness=50, correlation='noise'):
    """

    Parameters
    ----------
    date : Date object
    cs : str
        Stimulus name used for the correlation.
    visual_drivenness : float
        The amount to which a cell is visually driven
    correlation : str {'noise'}
        The correlation type used for clustering

    Returns
    -------
    array of floats
        Cluster number, with unclustered cells assigned np.nan

    """

    vdrive = driven_legacy.visually(date, 'plus') > visual_drivenness
    corr = correlations_legacy.noise(date, 'plus')
    ncells = np.shape(corr)[0]
    nc = flow.netcom.nxgraph(np.arange(ncells), corr, limits=vdrive).communities()

    return nc


def _reward_non(date, visual_drivenness=50, correlation='noise', reward_cells_required=1):
    """
    Generate arrays determining whether cells are in reward clusters,
    or non-reward clusters.

    Parameters
    ----------
    date : Date instance
    visual_drivenness : int
        The amount to which a cell is visually driven
    correlation : str {'noise'}
        The correlation type used for clustering
    reward_cells_required : int
        The number of food-cue-reward cells in a cluster required to define
        it as a reward cluster.

    Returns
    -------
    tuple of boolean arrays
        First element, True means cells are in non-reward clusters,
        Second, True means cells are in reward clusters
        (Cells can be false in both arrays)
    """

    vdrive = driven.visually(date, 'plus') > visual_drivenness
    corr = correlations.noise(date, 'plus')
    ncells = np.shape(corr)[0]

    nc = flow.netcom.nxgraph(np.arange(ncells), corr, limits=vdrive).communities()
    lbls = flow.categories.labels(date,
                                  categories=('lick', 'reward', 'non'),
                                  cluster_numbers=nc)

    non, rew = np.zeros(ncells) > 1, np.zeros(ncells) > 1
    for i, lbl in enumerate(lbls):
        if lbl == 'non':
            non[i] = True
        elif lbl == 'reward':
            rew[i] = True

    return non, rew