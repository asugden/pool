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

    df['fcreward'] = calc_legacy.glm_groups.fraction(run.parent, 'ensure-vdrive-plus')
    df['reward'] = calc_legacy.glm_groups.fraction(run.parent, 'ensure')

    return df


def dataframe(sorter, run_function, event_threshold=0.1, legacy=True, per_cell=False,
                    deconvolved_threshold=0.2,
                    trange=(-2, 3)):
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
            'dprime_pool': [calc.performance.dprime(run.parent)],
        }

        if legacy == True:
            data['dprime_legacy'] = [calc_legacy.performance.dprime(run.parent)]

            for cs in config.stimuli():
                data['react_%s'%cs] = [nannone(calc_legacy.reactivation_rate.freq(run.parent, cs))]

        for cs in config.stimuli():
            if legacy:
                evs = calc_legacy.reactivation_rate.events(run, run.run, cs)
            else:
                evs = calc.reactivation_rate.events(run, cs, event_threshold)

            if evs is not None and len(evs) > 0:
                csdata = deepcopy(data)
                for key in csdata:
                    csdata[key] = csdata[key]*len(evs)
                csdata['frame'] = evs
                csdata['stimulus'] = [cs]*len(evs)

                csdata['reward_rich'] = np.zeros(len(evs)) > 1
                csdata['reward_poor'] = np.zeros(len(evs)) > 1
                csdata['reward_mixed'] = np.zeros(len(evs)) > 1

                if cs == 'plus':
                    if legacy:
                        rewnon = calc_legacy.reactivation_rate.event_rich_poor(run, run.run)
                    else:
                        rewnon = calc.reactivation_rate.event_rich_poor(run, cs, event_threshold)
                    rewnon = np.array(rewnon) if rewnon is not None else [np.nan]*len(evs)

                    csdata['rewnon'] = rewnon
                    csdata['reward_rich'][rewnon >= 0.8] = True
                    csdata['reward_poor'][rewnon <= 0.2] = True
                    csdata['reward_mixed'][(0.2 < rewnon) & (rewnon < 0.8)] = True

                default = pd.DataFrame(csdata)
                date_df = run_function(run, default)

                if per_cell:
                    old_date_df = date_df
                    date_df = None

                    t2p = run.trace2p()
                    trs = t2p.trace('deconvolved')

                    ncells = np.arange(np.shape(trs)[0])
                    celldata = {}
                    for ev in evs:
                        if -1*trange[0] < ev < np.shape(trs)[1] - trange[1]:
                            act = np.nanmax(trs[:, ev + trange[0]:ev + trange[1]], axis=1)
                            act = act > deconvolved_threshold
                            celldata['cell_id'] = ncells[act]

                            if len(ncells[act]) > 0:
                                cell_df = pd.DataFrame(celldata)
                                ind = old_date_df.index[old_date_df.frame == ev][0]
                                for key in old_date_df.keys():
                                    cell_df[key] = old_date_df.at[ind, key]

                                if date_df is None:
                                    date_df = cell_df
                                else:
                                    date_df = pd.concat([date_df, cell_df], ignore_index=True, sort=True)

                if date_df is not None:
                    if df is None:
                        df = date_df
                    else:
                        df = pd.concat([df, date_df], ignore_index=True, sort=True)

    return df
