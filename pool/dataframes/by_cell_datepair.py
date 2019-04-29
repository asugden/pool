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


# from lib import analysis
# from lib import glm
# # from lib import grapher
# from lib import labels
# from lib import metadata
# from lib import netcom
# from lib import parseargv
# from lib import xday

# import fd_xday_extras as xtra


def getanalyses(andb, mouse, date, categorize, analyses):
    """
    Get a list of analyses and add them to a dataframe.

    Parameters
    ----------
    andb : analysis database object
    mouse : str
    date : str
    categorize : list of strings
        A list of labels to categorize by
    analyses : list of strings
        A list of analyses to add to the pandas dataframe.

    Returns
    -------
    Pandas dataframe
        A dataframe containing all cells, labeled, with all analyses

    """

    lbls1 = labels.categorize(mouse, date, categorize, False, (None, None), andb=andb)
    df = pd.DataFrame(lbls1, columns=['label'])
    df['mouse'] = mouse
    df['date'] = date

    for an in analyses:
        df[an.replace('-0.1', '').replace('-', '_')] = andb.get(an, mouse, date)

    df['base_lbl'] = labels.categorize(
        mouse, date, ['lick', 'ensure', 'reward-cluster-1', 'reward-cluster-non-1'],
        False, (None, None), andb=andb)
    df['nolick_lbl'] = labels.categorize(
        mouse, date, ['lick', 'ensure', 'reward-nolick-cluster-1', 'reward-nolick-cluster-non-1'],
        False, (None, None), andb=andb)
    df.loc[df['base_lbl'] == 'reward-cluster-1', 'base_lbl'] = 'reward'
    df.loc[df['nolick_lbl'] == 'reward-nolick-cluster-1', 'nolick_lbl'] = 'reward'
    df.loc[df['base_lbl'] == 'reward-cluster-non-1', 'base_lbl'] = 'non'
    df.loc[df['nolick_lbl'] == 'reward-nolick-cluster-non-1', 'nolick_lbl'] = 'non'

    return df






def addglm(df, mouse, date):
    """
    Add GLM components to an existing dataframe for a day.

    Parameters
    ----------
    df : Pandas dataframe
    mouse : str
    date : str

    Returns
    -------
    Pandas dataframe
        Updated to include glm components.

    """

    beh = glm.glm(mouse, date)
    expl = beh.explained()
    for cellgroup in expl:
        df['deviance_explained_%s' % cellgroup] = expl[cellgroup]
    return df


def addcluster(df, andb, mouse, date):
    """

    Parameters
    ----------
    df
    andb
    mouse
    date

    Returns
    -------

    """

    gr = netcom.graph(andb, 'plus', vdrive=50)
    coms = gr.communities() + 1
    coms[np.invert(np.isfinite(coms))] = 0
    df['cluster_number'] = 100*int(date) + coms
    return df


def addxday(day1md, day2md, day1df, day2df):
    """
    Combine two dataframes across a pair of days with xday alignment.

    Parameters
    ----------
    day1md : tuple
        Day 1 metadata (mouse, reversal pre (0) or post (1), date)
    day2md : tuple
        Day 2 metadata (mouse, reversal pre (0) or post (1), date)
    day1df : Pandas dataframe
        Day 1 populated dataframe
    day2df : Pandas dataframe
        Day 2 populated dataframe

    Returns
    -------
    Pandas dataframe
        Combined dataframe of pair-day
    """

    ids1, ids2 = xday.ids(day1md[0], day1md[-1], day2md[-1])
    dfids2 = np.array([np.nan]*len(day2df))
    dfids2[ids2] = ids1

    day1df['xday_ids'] = day1df.index
    day2df['xday_ids'] = dfids2

    return day1df.copy(), day2df.copy()


def addpairs(df, andb, mouse, date, analyses):
    """
    Get a list of analyses and add them to a dataframe.

    Parameters
    ----------
    df
    andb : analysis database object
    mouse : str
    date : str

    Returns
    -------
    Pandas dataframe
        A dataframe containing all cells, labeled, with all analyses

    """

    idx = pd.MultiIndex.from_product([df.xday_ids, df.xday_ids],
                                     names=('cell1_id', 'cell2_id'))

    # Add analyses
    data, cols, unfound = [], [], []
    for an in analyses:
        andata = andb.get(an, mouse, date)
        if andata is not None:
            data.append(andata.flatten())
            cols.append(an.replace('-0.1', '').replace('-', '_'))
        else:
            unfound.append(an.replace('-0.1', '').replace('-', '_'))

    # Join the pair-wise analysis database with the cell-wise database
    df['cell1_id'] = df.xday_ids
    multidf = pd.DataFrame(data=np.array(data).T, index=idx, columns=cols)
    for unf in unfound:
        multidf[unf] = np.nan

    pairdf = pd.merge(df[np.isfinite(df['cell1_id'])], multidf.reset_index(),
                      how='inner', on='cell1_id')

    pairdf = pairdf.loc[pairdf['cell2_id'] > pairdf['cell1_id'], :]

    # Add in the cell2 information
    df['cell2_id'] = df.xday_ids
    pairdf = pd.merge(pairdf, df[np.isfinite(df['cell2_id'])].add_suffix('_cell2'),
                      how='inner', left_on='cell2_id', right_on='cell2_id_cell2')

    return pairdf


def combinexday(day1md, day2md, day1df, day2df, pairs=False):
    """
    Combine two dataframes across a pair of days with xday alignment.

    Parameters
    ----------
    day1md : tuple
        Day 1 metadata (mouse, reversal pre (0) or post (1), date)
    day2md : tuple
        Day 2 metadata (mouse, reversal pre (0) or post (1), date)
    day1df : Pandas dataframe
        Day 1 populated dataframe
    day2df : Pandas dataframe
        Day 2 populated dataframe
    pairs : bool
        If true, combine as pairs of cells rather than individual cells

    Returns
    -------
    Pandas dataframe
        Combined dataframe of pair-day
    """

    ans = [key for key in day1df.keys() if np.issubdtype(day1df[key].dtype, np.number)]

    day1df = day1df.add_suffix('_day1')
    day2df = day2df.add_suffix('_day2')

    if pairs:
        pairdf = pd.merge(day1df, day2df, how='inner', left_on=['cell1_id_day1', 'cell2_id_day1'],
                          right_on=['cell1_id_day2', 'cell2_id_day2'])
    else:
        pairdf = pd.merge(day1df, day2df, how='inner', left_on='xday_ids_day1', right_on='xday_ids_day2')

    for an in ans:
        pairdf['d_%s' % an] = pairdf['%s_day2' % an] - pairdf['%s_day1' % an]

    return pairdf


def main(clargs):
    # Set the default parameters
    defaults = {
        'cell-threshold': 50,  # mutual information or visual driven cutoff, recommend 0.01 and 80, respectively
        'day-distance': 6,  # max distance of days, -1 allows any distance

        'categorize': ['lick', 'ensure', 'quinine', 'plus-only', 'minus-only', 'neutral-only'],  # Category order to
        # assign
        'propagate-labels': (None, None),  # Propagate labels in time
        'drop-licks': False,  # Remove cells with licking from all analyses

        'xday': True,
        'pairs': False,
        'pre-apply-threshold': False,
    }

    lpars = parseargv.extractkv(clargs, defaults)
    analyses = [
        'graph-clustering-%s',
        # 'graph-clustering-nolick-%s',
        # 'clustspont-%s',
        'repcount-0.1-%s',
        'visually-driven-%s',
        # 'stimulus-dff-0-2-%s',
        # 'stimulus-dff-2-4-%s',
        # 'stimulus-dff-all-0-2-%s',
        # 'stimulus-dff-all-2-4-%s',
        # 'stimulus-decon-2-4-%s',
        'hmm-behavior-%s',
        # (1.5, 2.5), (1.7, 4), (2.3, 3.3), (2.5, 3.5)
    ]
    ans = [v%cs for v in analyses for cs in ['plus', 'neutral', 'minus']] + \
        ['reward-specific-replay-plus',
         # 'stim-ratio-decon-0-2-4-plus',
         # 'stim-ratio-decon-0-2-4-correct-plus',
         # 'stim-ratio-decon-0-2-4-pav-plus',
         # 'stim-ratio-decon-0-2-reward-plus',
         # 'stim-ratio-dff-0-2-4-plus',
         # 'stim-ratio-dff-0-2-4-correct-plus',
         # 'stim-ratio-dff-0-2-4-pav-plus',
         # 'stim-ratio-dff-0-2-reward-plus',
         # 'stim-ratio-dff-tau-0-2-4-plus',
         # 'stim-ratio-dff-tau-0-2-4-correct-plus',
         # 'stim-ratio-dff-tau-0-2-4-pav-plus',
         # # 'stim-com-0-2-plus',
         # # 'stim-com-0-2-correct-plus',
         # # 'stim-com-0-2-pav-plus',
         # # 'stim-com-0-4-plus',
         # # 'stim-com-0-4-correct-plus',
         # # 'stim-com-0-4-pav-plus',
         # 'fano-factor-plus',
         # 'fano-factor-decon-plus',
         # # 'reward-nonspecific-replay-plus',
         'nonreward-specific-replay-plus',
         # # 'nonreward-nonspecific-replay-plus',
         # # 'chprob-gopos-nolick-plus',
         'hmm-dprime',
         'hmm-criterion',
         'hmm-engagement',
         # 'reward-specific-repfreq-plus',
         # 'nonreward-specific-repfreq-plus',
         # 'mean-lick-latency-plus',
         # 'mean-lick-latency-plus-stim',
         # 'mean-lick-latency-hit',
         # 'mean-lick-latency-go',
         # 'mean-lick-latency-go-stim',
         # 'stim-dff-0-2-plus',
         # 'stim-decon-0-2-plus',
         # 'stim-dff-reward-plus',
         # 'stim-decon-reward-plus',
         # 'replay-freq-0.1-plus',
         ]

    pair_analyses = [
        'reppair-0.1-%s',
        'noise-correlation-%s',
        # 'noise-correlation-nolick-%s',
    ]
    pair_analyses = [v%cs for v in pair_analyses for cs in ['plus', 'neutral', 'minus']] + \
                    [# 'spontcorr-allall',
                     'spontaneous-correlation',]
                     # 'reward-specific-replay-pair-plus',
                     # 'nonreward-specific-replay-pair-plus',
                     # 'reward-nonspecific-replay-pair-plus',
                     # 'nonreward-nonspecific-replay-pair-plus',]

    andb = analysis.db()
    days = parseargv.sorteddays(clargs, classifier=False, trace=False, force=True)
    lastmd, lastdf, df = ('mouse', 'reversal', '180902'), None, None
    while days.next():
        md, args = days.get()
        andb.md(md[0], md[1])
        rev = 0 if int(metadata.reversal(md[0])) < int(md[1]) else 1
        print(md[0], md[1])

        nextdf = getanalyses(andb, md[0], md[1], lpars['categorize'], ans)
        nextdf = addglm(nextdf, md[0], md[1])
        nextdf = addcluster(nextdf, andb, md[0], md[1])

        if not lpars['xday']:
            df = pd.concat([df, nextdf], ignore_index=True)

        else:
            nextmd = (md[0], rev, md[1])

            tdelta = datetime.strptime(str(nextmd[2]), '%y%m%d') - datetime.strptime(str(lastmd[2]), '%y%m%d')
            if nextmd[:-1] == lastmd[:-1] and tdelta.days <= lpars['day-distance']:
                xlastdf, xnextdf = addxday(lastmd, nextmd, lastdf, nextdf)

                if lpars['pairs']:
                    xlastdf = addpairs(xlastdf, andb, lastmd[0], lastmd[-1], pair_analyses)
                    xnextdf = addpairs(xnextdf, andb, nextmd[0], nextmd[-1], pair_analyses)

                pairdf = combinexday(lastmd, nextmd, xlastdf, xnextdf, lpars['pairs'])

                if lpars['pre-apply-threshold']:
                    pairdf = pairdf.loc[((df['visually_driven_%s_day1'%'plus'] >= lpars['cell-threshold']) &
                                         (df['visually_driven_%s_day2'%'plus'] >= lpars['cell-threshold'])) |
                                        ((df['visually_driven_%s_day1'%'neutral'] >= lpars['cell-threshold']) &
                                         (df['visually_driven_%s_day2'%'neutral'] >= lpars['cell-threshold'])) |
                                        ((df['visually_driven_%s_day1'%'minus'] >= lpars['cell-threshold']) &
                                         (df['visually_driven_%s_day2'%'minus'] >= lpars['cell-threshold'])), :]

                pairdf['day_distance'] = tdelta.days
                df = pd.concat([df, pairdf], ignore_index=True)

            lastmd = nextmd
            lastdf = nextdf

    df.fillna(value=np.nan, inplace=True)

    if not lpars['xday']:
        df['date'] = pd.to_numeric(df['date'])
    else:
        df['date_day1'] = pd.to_numeric(df['date_day1'])
        df['date_day2'] = pd.to_numeric(df['date_day2'])

    # Remove licking cells
    if lpars['drop-licks']:
        df = df.loc['label_day1' != 'lick', :]

    # Make a data subset for graphing
    sdf = xtra.SubData(df, xday=lpars['xday'], pairs=lpars['pairs'])

    # Make interactive
    code.interact(local=dict(globals(), **locals()))

def parse_args():
    """Add arguments to be parsed. Uses default parser based on mice, tags, and dates."""
    arg_parser = flow.misc.default_parser(
        description="""
        Plot trial responses of pairs of cells that match criteria.""",
        arguments=('mice', 'tags', 'dates'))
    arg_parser.add_argument(
        '-x', '--xday', action="store_false",  # Default is true
        help='If true, keep only cells found across pairs of days.')
    arg_parser.add_argument(
        '-p', '--pairs', action="store_true",  # Default is false
        help='If true, each pandas entry will be a pair of cells rather than a cell.')

    arg_parser.add_argument(
        '-v', '--visually_driven', type=int, default=50,
        help='The visual-drivenness threshold. Standard is 50.')
    arg_parser.add_argument(
        '-c', '--categorize', nargs='+', type=str, default=('lick', 'ensure', 'quinine',
                                                            'plus-only', 'minus-only', 'neutral-only'),
        help='Order in which to categorize cells by their labels.')
    arg_parser.add_argument(
        '-s', '--save_path', type=str, default=paths.graphcrossday(),
        help='Directory in which to save any graphs.')

    # Rarely used options
    arg_parser.add_argument(
        '-D', '--day_distance', nargs=2, type=int, default=(0, 6),
        help='Distance between days, inclusive.')
    arg_parser.add_argument(
        '-S', '--sequential', action="store_false",  # Default is true
        help='Limit only to sequential recording days.')
    arg_parser.add_argument(
        '-R', '--cross_reversal', action="store_true",  # Default is false
        help='Allow day pairs across reversal if true.')

    args = arg_parser.parse_args()

    return args


def main(args):
    """Main function."""

    df = None
    if args.xday:
        sorter = sorters.DatePairSorter.frommeta(
            mice=args.mice, dates=args.dates, day_distance=args.day_distance, sequential=args.sequential,
            cross_reversal=args.cross_reversal, tags=args.tags)

        for day1, day2 in sorter:
            day1df = date_analyses(day1)
            day2df = date_analyses(day2)

            # nextdf = getanalyses(andb, md[0], md[1], lpars['categorize'], ans)
            # nextdf = addglm(nextdf, md[0], md[1])
            # nextdf = addcluster(nextdf, andb, md[0], md[1])
            #
            # if lpars['pairs']:
            #     xlastdf = addpairs(xlastdf, andb, lastmd[0], lastmd[-1], pair_analyses)
            #     xnextdf = addpairs(xnextdf, andb, nextmd[0], nextmd[-1], pair_analyses)
            #
            # pairdf = combinexday(lastmd, nextmd, xlastdf, xnextdf, lpars['pairs'])
            # pairdf['day_distance'] = tdelta.days

            day1df = day1df.add_suffix('_day1')
            day2df = day2df.add_suffix('_day2')

            # if pairs:
            #     pairdf = pd.merge(day1df, day2df, how='inner', left_on=['cell1_id_day1', 'cell2_id_day1'],
            #                       right_on=['cell1_id_day2', 'cell2_id_day2'])
            # else:
            #     pairdf = pd.merge(day1df, day2df, how='inner', left_on='xday_ids_day1', right_on='xday_ids_day2')

            pairdf = pd.concat([day1df, day2df], axis=1)

            df = pd.concat([df, pairdf], ignore_index=True, sort=True)
    else:
        sorter = sorters.DateSorter.frommeta(
            mice=args.mice, dates=args.dates, tags=args.tags)

        for date in sorter:
            date_df = date_analyses(date)

            df = pd.concat([df, date_df], ignore_index=True, sort=True)

    return df


def jupyter(mice=None, dates=None, tags=None, xday=True):
    """
    Run from Jupyter Notebook

    Parameters
    ----------
    mice
    dates
    tags

    Returns
    -------

    """

    class TempArgs:
        def __init__(self):
            self.mice = mice
            self.dates = dates
            self.tags = tags
            self.xday = xday
            self.pairs = False
            self.visually_driven = 50
            self.categorize = ('lick', 'ensure', 'quinine', 'plus-only', 'minus-only', 'neutral-only')
            self.save_path = paths.graphcrossday()
            self.day_distance = (0, 6)
            self.sequential = True
            self.cross_reversal = False

    args = TempArgs()
    return main(args)






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
        vdrive_classic = calc.driven.visually_classic(date, cs)
        vdrive_legacy = calc_legacy.driven.visually(date, cs)

        if df is None:
            df = pd.DataFrame(np.array([vdrive_classic, vdrive_legacy]).transpose(),
                              columns=['vdrive_classic_%s'%cs, 'vdrive_legacy_%s'%cs])
        else:
            df['vdrive_classic_%s'%cs] = calc.driven.visually_classic(date, cs)
            df['vdrive_legacy_%s'%cs] = calc_legacy.driven.visually(date, cs)

        df['react_%s'%cs] = calc_legacy.reactivation_rate.cell(date, cs)
        df['connectivity_%s'%cs] = calc_legacy.connectivity.total(date, cs)

        df[cs] = calc.glm_groups.responsive(date, cs)

    df['mouse'] = date.mouse
    df['date'] = date.date
    df['reversed'] = flow.metadata.reversal(date.mouse) < date.date
    df['dprime_legacy'] = calc_legacy.performance.dprime(date)
    df['dprime_pool'] = calc.performance.dprime(date)

    df['lick'] = calc.glm_groups.responsive(date, 'lick')
    df['ensure'] = calc.glm_groups.responsive(date, 'ensure')
    df['quinine'] = calc.glm_groups.responsive(date, 'quinine')

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


if __name__ == '__main__':
    clargs = parse_args()
    main(clargs)
