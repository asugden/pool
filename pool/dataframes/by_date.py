import pandas as pd

import flow.grapher
import flow.metadata
import flow.misc
import flow.paths
import flow.sorters
from flow.misc.type_conversion import nannone
from .. import calc
from .. import calc_legacy


def example_date_function(date, df):
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

    df['cossim_plus'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure'))
    df['cossim_plus_dropzero'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', drop_glm_zeros=True, rectify=True))
    df['cossim_plus_trials_max'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', compare_to_trials=True, max_across_trial=True, rectify=True))
    df['cossim_plus_trials'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', compare_to_trials=True, rectify=True))

    df['fcreward'] = calc_legacy.glm_groups.fraction(date, 'ensure-vdrive-plus')
    df['reward'] = calc_legacy.glm_groups.fraction(date, 'ensure')

    return df


def dataframe(sorter, date_function):
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
    for date in sorter:
        data = {
            'mouse': [date.mouse],
            'date': [date.date],
            'reversed': [flow.metadata.reversal(date.mouse) < date.date],
            'dprime_legacy': [calc_legacy.performance.dprime(date)],
            'dprime_pool': [calc.performance.dprime(date)],
            'react_plus': [nannone(calc_legacy.reactivation_rate.freq(date, 'plus'))],
            'react_neutral': [nannone(calc_legacy.reactivation_rate.freq(date, 'neutral'))],
            'react_minus': [nannone(calc_legacy.reactivation_rate.freq(date, 'minus'))],
            'react_hungry_plus': [nannone(calc_legacy.reactivation_rate.freq(date, 'plus', '-hungry'))],
            'react_hungry_neutral': [nannone(calc_legacy.reactivation_rate.freq(date, 'neutral', '-hungry'))],
            'react_hungry_minus': [nannone(calc_legacy.reactivation_rate.freq(date, 'minus', '-hungry'))],
        }

        default = pd.DataFrame(data)
        date_df = date_function(date, default)

        if df is None:
            df = date_df
        else:
            df = pd.concat([df, date_df], ignore_index=True, sort=True)

    return df
