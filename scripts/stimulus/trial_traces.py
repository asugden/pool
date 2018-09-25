"""Look at individual trial stimulus responses from example cells."""
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')

import itertools as it
from operator import itemgetter
import os
import pandas as pd

import flow
from flow import misc
import flow.metadata as metadata
import pool
from pool.layouts import stimulus as pls


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Script to plot mean stimulus response over days.""",
        epilog="""
        This is the epilog.
        """, arguments=('mice', 'tags', 'dates', 'overwrite'))
    arg_parser.add_argument(
        "-T", "--trace_type", choices=('dff', 'deconvolved', 'raw'), default="dff",
        help="Trace type to plot.")
    arg_parser.add_argument(
        "-R", "--t_range_s", nargs=2, type=int, default=(-1, 8),
        help="Time range around stimulus to plot.")
    arg_parser.add_argument(
        "-b", "--baseline", nargs=2, type=int, default=None,
        help='Baseline used for dFF trace.')
    arg_parser.add_argument(
        "-e", "--errortrials", choices=(-1, 0, 1, 2), type=int, default=-1,
        help="-1 is off, 0 is correct trials, 1 is error trials, 2 is diff of error trials.")
    arg_parser.add_argument(
        "-N", "--normalize", action="store_true",
        help="If True, normalize each individual trace by z-scoring it.")
    arg_parser.add_argument(
        "-M", "--maxrois", action="store", type=int, default=10,
        help="Pull out the 'maxrois' top ROIs per stim type.")

    args = arg_parser.parse_args()

    return args


def roi_df(date):
    """Return a dataframe of all ROIs with sorting order and labels added.

    Would eventually like to pull this out, but not exactly sure where it
    should go.

    """
    adb = pool.database.db()
    sort_order = adb.get('sort_order', date.mouse, date.date)
    sort_borders = adb.get('sort_borders', date.mouse, date.date)
    df = pd.DataFrame(
        {'sort_idx': range(len(sort_order)), 'roi_idx': sort_order})
    for group, first_idx in sorted(sort_borders.items(), key=itemgetter(1)):
        df.loc[df.index[first_idx:], 'sort_label'] = group
    return df.set_index('roi_idx').sort_index()


def traces_per_cell(
        date, trace_type, t_range_s, errortrials, baseline,
        max_rois_per_group=-1, normalize=False):
    """Plot traces for each Date."""
    rois = roi_df(date)
    if max_rois_per_group > 0:
        rois = rois.groupby('sort_label').sort_idx.nsmallest(
            max_rois_per_group).reset_index().set_index('roi_idx')
    for _, (roi_idx, group) in rois.reset_index().sort_values(
            'sort_idx')[['roi_idx', 'sort_label']].iterrows():
        fig = pls.trial_traces(
            date, roi_idx, t_range_s=t_range_s, trace_type=trace_type,
            errortrials=errortrials, baseline=baseline, normalize=normalize,
            fig_kw={'figsize': (16, 9)})
        fig.suptitle(
            '{} - {} - {} - {}'.format(
                date.mouse, date.date, roi_idx, group))
        yield(fig)


def main():
    """Main function."""
    filename = '{}_trial_traces.pdf'
    save_dir = os.path.join(flow.paths.graphd, 'trial_traces')
    args = parse_args()
    sorter = metadata.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    for date in sorter:
        save_path = os.path.join(save_dir, date.mouse, filename.format(date))

        # Make this script cron-able
        if not args.overwrite and os.path.exists(save_path):
            continue

        figs = traces_per_cell(
            date, args.trace_type, args.t_range_s, args.errortrials,
            args.baseline, args.maxrois, args.normalize)
        summary_fig = misc.summary_page(sorter, figsize=(16, 9), **vars(args))
        misc.save_figs(save_path, it.chain([summary_fig], figs))
        print(save_path)


if __name__ == '__main__':
    main()
