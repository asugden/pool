"""Look at stimulus responses."""
from __future__ import print_function
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')

import os

import flow
from flow import misc

from pool.layouts import stimulus as pls


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Script to plot mean stimulus response over days.""",
        epilog="""
        The 'dates' option really only makes sense with a single Mouse.
        """, arguments=('mice', 'tags', 'dates', 'overwrite', 'verbose'))
    arg_parser.add_argument(
        "-T", "--trace_type", choices=('dff', 'deconvolved', 'raw'), default="dff",
        help="Trace type to plot.")
    arg_parser.add_argument(
        "-R", "--t_range_s", nargs=2, type=int, default=(-2, 8),
        help="Time range around stimulus to plot.")
    arg_parser.add_argument(
        "-b", "--baseline", nargs=2, type=int, default=(-1, 0),
        help='Baseline used for dFF trace.')
    arg_parser.add_argument(
        "-e", "--errortrials", choices=(-1, 0, 1, 2), type=int, default=-1,
        help="-1 is off, 0 is correct trials, 1 is error trials, 2 is diff of error trials.")
    # arg_parser.add_argument(
    #     "-H", "--hungry_sated", choices=(0, 1, 2), type=int, default=0,
    #     help="0 is hungry trials, 1 is sated trials, 2 is hungry-sated")

    args = arg_parser.parse_args()

    return args


def main():
    """Main function."""
    filename = '{}_stimulus_response.pdf'
    save_dir = os.path.join(flow.paths.graphd, 'stimulus_response')
    args = parse_args()
    sorter = flow.MouseSorter.frommeta(
        mice=args.mice, tags=args.tags)
    for mouse in sorter:
        save_path = os.path.join(save_dir, filename.format(mouse))
        if not args.overwrite and os.path.exists(save_path):
            continue

        if args.verbose:
            print('Generating stimulus response plots: {}'.format(mouse))

        fig = pls.stimulus_response(
            mouse.dates(dates=args.dates, tags=args.tags),
            t_range_s=args.t_range_s, trace_type=args.trace_type,
            errortrials=args.errortrials, baseline=args.baseline, sharey=True)
        summary_fig = misc.summary_page(
            mouse.dates(), figsize=(16, 9), **vars(args))
        misc.save_figs(save_path, [summary_fig, fig])
        print(save_path)


if __name__ == '__main__':
    main()
