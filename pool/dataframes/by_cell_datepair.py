import numpy as np
import pandas as pd

# import flow.misc
import flow.metadata
from flow import glm
from flow import sorters
from flow.misc import math
from .. import calc
from .. import calc_legacy
from .. import config


def example_cell_function(date, df):
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

    df['reward'] = calc.clusters.reward(date)
    df['nonreward'] = calc.clusters.nonreward(date)

    return df


def date_analyses(date):
    """
    Get a list of analyses and add them to a dataframe.

    Parameters
    ----------
    date : Date object

    Returns
    -------
    Pandas dataframe
        A dataframe containing all cells, labeled, with all analyses

    """

    df = None
    for cs in config.stimuli():  # pool.config.stimuli():
        # vdrive_classic = calc.driven.visually_classic(date, cs)
        vdrive_legacy = calc_legacy.driven.visually(date, cs)

        if df is None:
            df = pd.DataFrame(np.array([vdrive_legacy]).transpose(),
                              columns=['vdrive_legacy_%s'%cs])
        else:
            # df['vdrive_classic_%s'%cs] = calc.driven.visually_classic(date, cs)
            df['vdrive_legacy_%s'%cs] = calc_legacy.driven.visually(date, cs)

        df['react_%s'%cs] = calc_legacy.reactivation_rate.cell(date, cs)
        df['connectivity_%s'%cs] = calc_legacy.connectivity.total(date, cs)

        # df[cs] = calc.glm_groups.responsive(date, cs)

    df['mouse'] = date.mouse
    df['date'] = date.date
    df['reversed'] = flow.metadata.reversal(date.mouse) < date.date
    df['dprime_legacy'] = calc_legacy.performance.dprime(date)
    df['cell_id'] = date.cells
    # df['dprime_pool'] = calc.performance.dprime(date)

    # df['lick'] = calc.glm_groups.responsive(date, 'lick')
    # df['ensure'] = calc.glm_groups.responsive(date, 'ensure')
    # df['quinine'] = calc.glm_groups.responsive(date, 'quinine')

    lbls = flow.glm.labels(date.mouse, date.date)

    return df


def dataframe(sorter, cell_function):
    """
    Iterates over sorter, running date_function and
    adding those parameters to default mouse, date, and dprime.

    Parameters
    ----------
    sorter : Sorter
        To iterate over
    cell_function : function with date and dataframe as arguments
        Append to and return a per-cell dataframe from a date
        that will be concatenated.

    Returns
    -------
    dataframe

    """

    df = None

    for day1, day2 in sorter:  # Include metadata like day distance
        day1df = date_analyses(day1)
        day2df = date_analyses(day2)

        day1df = cell_function(day1, day1df)
        day2df = cell_function(day2, day2df)

        keys = day1df.keys()

        day1df = day1df.add_suffix('_day1')
        day2df = day2df.add_suffix('_day2')

        pairdf = pd.concat([day1df, day2df], axis=1)

        for key in keys:
            if np.issubdtype(pairdf['%s_day1'%key].dtype, np.number):
                pairdf['d_%s'%key] = pairdf['%s_day2'%key] - pairdf['%s_day1'%key]

        df = pd.concat([df, pairdf], ignore_index=True, sort=True)

    df.fillna(value=np.nan, inplace=True)
    return df
