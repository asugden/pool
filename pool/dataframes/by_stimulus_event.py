from copy import deepcopy
import numpy as np
import pandas as pd

import flow.grapher
import flow.metadata
import flow.misc
import flow.paths
import flow.sorters
from flow.misc.type_conversion import nannone
from .. import calc
from .. import calc_legacy
from .. import config


def example_run_function(run, df):
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

    df['fcreward'] = calc.glm_groups.fraction(run.parent, 'ensure-vdrive-plus')
    df['reward'] = calc.glm_groups.fraction(run.parent, 'ensure')

    return df


def dataframe(sorter, run_function):
    """
    Iterates over sorter, running date_function and
    adding those parameters to default mouse, date, and dprime.

    Parameters
    ----------
    sorter : Sorter
        To iterate over
    date_function : function with date and dataframe as arguments
        Append to and return a dataframe from a date
        that will be concatenated.

    Returns
    -------
    dataframe

    """

    df = None
    for run in sorter:
        data = {
            'mouse': [run.mouse],
            'date': [run.date],
            'run': [run.run],
            'reversed': [flow.metadata.reversal(run.mouse) < run.date],
            'dprime': [calc.performance.dprime(run.parent)],
        }

        t2p = run.trace2p()
        for cs in config.stimuli():
            for err in [0, 1]:
                evs = t2p.csonsets(cs, errortrials=err)
                csdata = deepcopy(data)
                for key in csdata:
                    csdata[key] = csdata[key]*len(evs)

                csdata['frame'] = evs
                csdata['stimulus'] = [cs]*len(evs)
                csdata['error'] = [err]*len(evs)

                default = pd.DataFrame(csdata)
                date_df = run_function(run, default)

                if df is None:
                    df = date_df
                else:
                    df = pd.concat([df, date_df], ignore_index=True, sort=True)

    return df
