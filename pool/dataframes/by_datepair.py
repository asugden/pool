from datetime import datetime
import numpy as np
import pandas as pd

import flow.grapher
import flow.metadata
import flow.misc
import flow.paths
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
    df['dprime'] = calc_legacy.performance.dprime(date)
    df['dprime_new'] = calc.performance.dprime(date)
    df['dprime_run'] = calc.performance.dprime(date, across_run=False)
    df['dprime_first'] = calc.performance.dprime_run(date.runs('training', tags='hungry')[0])
    df['dprime_last'] = calc.performance.dprime_run(date.runs('training', tags='hungry')[-1])
    df['engagement'] = calc.performance.engagement(date)
    df['reversed'] = flow.metadata.reversal(date.mouse) < date.date

    df['react_plus'] = nannone(calc_legacy.reactivation_rate.freq(date, 'plus'))
    df['react_neutral'] = nannone(calc_legacy.reactivation_rate.freq(date, 'neutral'))
    df['react_minus'] = nannone(calc_legacy.reactivation_rate.freq(date, 'minus'))

    # df['react_new_plus'] = nannone(calc.reactivation_rate.freq(date, 'plus'))
    # df['react_new_neutral'] = nannone(calc.reactivation_rate.freq(date, 'neutral'))
    # df['react_new_minus'] = nannone(calc.reactivation_rate.freq(date, 'minus'))

    # df['cosdist_plus'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'plus'))
    # df['cosdist_neutral'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'neutral'))
    # df['cosdist_minus'] = nannone(calc_legacy.cosine_distance.stimulus(date, 'minus'))

    # cosine_similarity_stimuli(
    #     date, cs, group, trace_type='deconvolved', start_s=0, end_s=1, trange_glm=(0, 1), rectify=False,
    #     exclude_outliers=False, remove_group=None, offset_glm_positive=False, remove_baseline_stimuli=False,
    #     drop_glm_zeros=False, max_across_trial=False, compare_to_trials=False, error_trials=0, binarize=None,
    #     correlation=False)

    df['cossim_plus'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure'))
    df['cossim_plus_dropzero'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', drop_glm_zeros=True, rectify=True))
    df['cossim_plus_trials_max'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', compare_to_trials=True, max_across_trial=True, rectify=True))
    df['cossim_plus_trials'] = nannone(calc.distance.cosine_similarity_stimuli(
        date, 'plus', 'ensure', compare_to_trials=True, rectify=True))

    return df


def main(args):
    """Main function."""
    sorter = flow.metadata.DatePairSorter.frommeta(
        mice=args.mice, dates=args.dates, day_distance=args.day_distance, sequential=args.sequential,
        cross_reversal=args.cross_reversal, tags=args.tags)
    # np.warnings.filterwarnings('ignore')

    df = None
    for day1, day2 in sorter:
        day1df = dataframe_date(day1)
        day2df = dataframe_date(day2)

        ans = [key for key in day1df.keys() if np.issubdtype(day1df[key].dtype, np.number)]

        day1df = day1df.add_suffix('_day1')
        day2df = day2df.add_suffix('_day2')
        pairdf = pd.concat([day1df, day2df], axis=1)

        tdelta = datetime.strptime(str(day2.date), '%y%m%d') - datetime.strptime(str(day1.date), '%y%m%d')
        pairdf['day_distance'] = tdelta.days

        for an in ans:
            pairdf['d_%s'%an] = pairdf['%s_day2'%an] - pairdf['%s_day1'%an]

        if df is None:
            df = pairdf
        else:
            df = pd.concat([df, pairdf], ignore_index=True, sort=True)

    return df


def subset_bidirectional_pairs(df):
    """
    Keep only those days in which there is a change in behavior that can be predicted from day1 or day2.
    Parameters
    ----------
    df

    Returns
    -------

    """


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
            self.xday = True
            self.pairs = False
            self.save_path = flow.paths.graphcrossday()
            self.day_distance = (0, 6)
            self.sequential = True
            self.cross_reversal = False

    # args = parse_args()
    args = TempArgs()
    return main(args)
