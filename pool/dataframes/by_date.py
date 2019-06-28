import datetime
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
from .. import database


def dataframe_date(date):
    """

    Parameters
    ----------
    date

    Returns
    -------

    """

    df = pd.DataFrame([date.mouse], columns=['mouse'])
    # df['mouse'] = date.mouse
    df['date'] = date.date
    df['date_obj'] = df['date'].apply(
        lambda x: datetime.datetime.strptime(str(x), '%y%m%d'))
    df['dprime'] = calc_legacy.performance.dprime(date)
    df['dprime_new'] = calc.performance.dprime(date)
    df['dprime_new_alltrials'] = calc.performance.dprime(
        date, hmm_engaged=False)
    df['dprime_run'] = calc.performance.dprime(date, across_run=False)
    df['dprime_day_start'] = calc.psytrack.dprime_day_start(date)
    df['dprime_day_end'] = calc.psytrack.dprime_day_end(date)
    df['d_dprime_day'] = calc.psytrack.d_dprime_day(date)
    df['reversed'] = flow.metadata.reversal(date.mouse) < date.date

    # df['react_plus'] = nannone(calc_legacy.reactivation_rate.freq(date, 'plus'))
    # df['react_neutral'] = nannone(calc_legacy.reactivation_rate.freq(date, 'neutral'))
    # df['react_minus'] = nannone(calc_legacy.reactivation_rate.freq(date, 'minus'))

    df['react_new_plus'] = nannone(calc.reactivation_rate.freq(date, 'plus'))
    df['react_new_neutral'] = nannone(calc.reactivation_rate.freq(date, 'neutral'))
    df['react_new_minus'] = nannone(calc.reactivation_rate.freq(date, 'minus'))
    df['react_new_comb'] = \
        df['react_new_plus'] + df['react_new_neutral'] + df['react_new_minus']
    df['react_all_plus'] = nannone(calc.reactivation_rate.freq(date, 'plus', state='all'))
    df['react_all_neutral'] = nannone(calc.reactivation_rate.freq(date, 'neutral', state='all'))
    df['react_all_minus'] = nannone(calc.reactivation_rate.freq(date, 'minus', state='all'))

    # df['cosdist_plus'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'plus'))
    # df['cosdist_neutral'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'neutral'))
    # df['cosdist_minus'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'minus'))

    # cosine_similarity_stimuli(
    #     date, cs, group, trace_type='deconvolved', start_s=0, end_s=1, trange_glm=(0, 1), rectify=False,
    #     exclude_outliers=False, remove_group=None, offset_glm_positive=False, remove_baseline_stimuli=False,
    #     drop_glm_zeros=False, max_across_trial=False, compare_to_trials=False, error_trials=0, binarize=None,
    #     correlation=False)



    return df


def date_fraction(df):
    """
    Add the fractional time passed for each mouse.

    Parameters
    ----------
    df : Dataframe

    Returns
    -------
    updated dataframe

    """

    for mouse in df['mouse'].unique():
        df.loc[df.mouse == mouse, 'sequential_date'], _ = \
            pd.factorize(df.loc[df.mouse == mouse, 'date'], sort=True)
        df.loc[df.mouse == mouse, 'fractional_date'] = \
            (df.loc[df.mouse == mouse, 'sequential_date'].astype(float)/
             df.loc[df.reversed == 0, 'sequential_date'].max())

    return df


def main(args):
    """Main function."""
    sorter = flow.sorters.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    # np.warnings.filterwarnings('ignore')

    df = None
    for date in sorter:
        df_date = dataframe_date(date)

        if df is None:
            df = df_date
        else:
            df = pd.concat([df, df_date], ignore_index=True, sort=True)

    df = date_fraction(df)

    return df


def parse_args():
    arg_parser = flow.misc.default_parser(
        description="""
        Plot clusters across pairs of days.""",
        arguments=('mice', 'tags', 'dates'))
    arg_parser.add_argument(
        '-p', '--save_path', type=str, default=flow.paths.graphcrossday(),
        help='The directory in which to save the output graph.')

    args = arg_parser.parse_args()

    return args


def jupyter(mice=None, dates=None, tags=None):
    """
    Pass the dataframe to a jupyter notebook.

    Parameters
    ----------
    mice : list of strings
        Mouse names
    dates : list of ints
        Dates
    tags : list of strings
        Tags

    Returns
    -------
    dataframe across days

    """

    class TempArgs:
        def __init__(self):
            self.mice = mice
            self.dates = dates
            self.tags = tags

    # args = parse_args()
    args = TempArgs()
    return main(args)


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

    df['fcreward'] = pool.calc_legacy.glm_groups.fraction(date, 'ensure-vdrive-plus')
    df['reward'] = pool.calc_legacy.glm_groups.fraction(date, 'ensure')

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
        }

        default = pd.DataFrame(data)
        date_df = date_function(date, default)

        if df is None:
            df = date_df
        else:
            df = pd.concat([df, date_df], ignore_index=True, sort=True)

    return df


if __name__ == '__main__':
    clargs = parse_args()
    main(clargs)
