import code
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from flow import glm
from flow import paths
from flow import sorters

# import flow.misc
import flow.metadata
from flow.misc import math
import flow.paths

from .. import config
from .. import calc
from .. import calc_legacy


def example_cellpair_function(date):
    """

    Parameters
    ----------
    date

    Returns
    -------

    """

    df = None
    for cs in config.stimuli():
        pair_rep_count = calc.reactivation_rate.pair(date, cs)
        if pair_rep_count is None:
            ncells = len(calc.clusters.reward(date))
            pair_rep_count = np.zeros((ncells, ncells))

        c1, c2, pcount = [], [], []
        for i in range(np.shape(pair_rep_count)[0]):
            for j in range(i + 1, np.shape(pair_rep_count)[1]):
                c1.append(i)
                c2.append(j)
                pcount.append(pair_rep_count[i, j])

        if df is None:
            df = pd.DataFrame(np.array([c1, c2, pcount]).transpose(),
                              columns=['cell1', 'cell2', 'pair_count_%s'%cs])
        else:
            df['pair_count_%s'%cs] = np.array(pcount)

    return df


def example_cell_function(date):
    """
    All date_functions must take date and df.

    Parameters
    ----------
    date : Date instance
    df : DataFrame

    Returns
    -------
    Updated DataFrame

    """

    df = None
    reward = calc.clusters.reward(date)
    non = calc.clusters.nonreward(date)
    cnum = calc.clusters.number_legacy(date, 'plus',
                                       visual_drivenness=50,
                                       correlation='noise')

    df = pd.DataFrame(np.array([np.arange(len(reward)), reward, non, cnum]).transpose(),
                      columns=['cell', 'reward', 'nonreward', 'cluster_number'])

    for cs in config.stimuli():  # pool.config.stimuli():
        df['vdrive_%s'%cs] = calc_legacy.driven.visually(date, cs)

    return df


def example_date_function(date, df):
    """
    An example function that takes into account wjetjer
    Parameters
    ----------
    date

    Returns
    -------

    """

    df['mouse'] = date.mouse
    df['date'] = date.date
    # df['reversed'] = flow.metadata.reversal(date.mouse) < date.date
    # df['dprime_legacy'] = calc_legacy.performance.dprime(date)
    # df['dprime_pool'] = calc.performance.dprime(date)

    return df


def dataframe(sorter, cellpair_function, cell_function=None,
              date_function=None, visually_driven=50):
    """
    Iterates over sorter, running date_function and
    adding those parameters to default mouse, date, and dprime.

    Parameters
    ----------
    sorter : Sorter
        To iterate over
    cellpair_function : function with date and dataframe as arguments
        Append to and return a per-cell dataframe from a date
        that will be concatenated.

    Returns
    -------
    dataframe

    """

    df = None
    for date in sorter:  # Include metadata like day distance
        pairdf = cellpair_function(date)

        if cell_function is not None:
            cell1df = cell_function(date)

            cell2df = cell1df.copy().add_suffix('_cell2')
            cell1df = cell1df.add_suffix('_cell1')

            pairdf = pairdf.merge(cell1df, left_on='cell1', right_on='cell_cell1')
            pairdf = pairdf.merge(cell2df, left_on='cell2', right_on='cell_cell2')

        if visually_driven > -1:
            pairdf['driven_cell1'] = pairdf[['vdrive_%s_cell1'%cs
                                             for cs in config.stimuli()]].max(axis=1)
            pairdf['driven_cell2'] = pairdf[['vdrive_%s_cell2'%cs
                                             for cs in config.stimuli()]].max(axis=1)

            pairdf = pairdf.loc[(pairdf['driven_cell1'] > visually_driven) &
                                (pairdf['driven_cell2'] > visually_driven), :]

        if date_function is not None:
            pairdf = date_function(date, pairdf)

        df = pd.concat([df, pairdf], ignore_index=True, sort=True)

    df.fillna(value=np.nan, inplace=True)
    return df
