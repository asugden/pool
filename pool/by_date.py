import numpy as np
import pandas as pd

import flow.grapher
import flow.metadata
import flow.misc
import flow.paths
import pool.calc.reactivation_rate
import pool.calc_legacy.performance
import pool.calc_legacy.reactivation_rate
import pool.database


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
    df['dprime'] = pool.calc_legacy.performance.dprime(date)

    df['react_plus'] = pool.calc_legacy.reactivation_rate.freq(date, 'plus')
    df['react_neutral'] = pool.calc_legacy.reactivation_rate.freq(date, 'neutral')
    df['react_minus'] = pool.calc_legacy.reactivation_rate.freq(date, 'minus')

    df['react_new_plus'] = pool.calc.reactivation_rate.freq(date, 'plus')
    df['react_new_neutral'] = pool.calc.reactivation_rate.freq(date, 'neutral')
    df['react_new_minus'] = pool.calc.reactivation_rate.freq(date, 'minus')

    return df


def main(args):
    """Main function."""
    sorter = flow.metadata.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    # np.warnings.filterwarnings('ignore')

    df = None
    for date in sorter:
        df_date = dataframe_date(date)

        if df is None:
            df = df_date
        else:
            df = pd.concat([df, df_date], ignore_index=True, sort=True)

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


if __name__ == '__main__':
    clargs = parse_args()
    main(clargs)
