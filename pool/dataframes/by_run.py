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


def dataframe_date(run):
    """

    Parameters
    ----------
    run

    Returns
    -------

    """

    df = pd.DataFrame([run.mouse], columns=['mouse'])
    df['mouse'] = run.mouse
    df['date'] = run.date
    df['run'] = run.run
    df['dprime'] = nannone(calc.performance.dprime_run(run, across_run=False))
    df['reversed'] = run.date < flow.metadata.reversal(run.mouse)

    df['cossim_plus'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'plus', 'ensure', rectify=False))
    df['cossim_neutral'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'neutral', 'ensure', rectify=False))
    df['cossim_minus'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'minus', 'ensure', rectify=False))

    df['cossim_plus_wide'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'plus', 'ensure', rectify=False, trange_glm=(-1, 2)))
    df['cossim_neutral_wide'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'neutral', 'ensure', rectify=False, trange_glm=(-1, 2)))
    df['cossim_minus_wide'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'minus', 'ensure', rectify=False, trange_glm=(-1, 2)))

    df['cossim_plus_0.5'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'plus', 'ensure', rectify=False, end_s=0.5))
    df['cossim_neutral_0.5'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'neutral', 'ensure', rectify=False, end_s=0.5))
    df['cossim_minus_0.5'] = nannone(calc.distance.cosine_similarity_stimuli_run(
        run, 'minus', 'ensure', rectify=False, end_s=0.5))

    return df


def main(args):
    """Main function."""

    if args.tags is None:
        args.tags = ['hungry']
    elif isinstance(args.tags, str):
        args.tags = [args.tags] + ['hungry']
    elif 'hungry' not in args.tags:
        args.tags = args.tags + ['hungry']

    sorter = flow.metadata.RunSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags, run_types='training')
    # np.warnings.filterwarnings('ignore')

    df = None
    for run in sorter:
        df_run = dataframe_date(run)

        if df is None:
            df = df_run
        else:
            df = pd.concat([df, df_run], ignore_index=True, sort=True)

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
