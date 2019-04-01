import numpy as np

import flow.categories
import flow.netcom
from . import correlations
from . import driven


def reward(date, visual_drivenness=50, correlation='noise', reward_cells_required=1):
    """

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
