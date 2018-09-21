"""Look at stimulus responses."""
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')

import os

import flow
from flow import misc
import flow.metadata2 as metadata

from pool.plotting import stimulus as pps


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Script to plot mean stimulus response over days.""",
        epilog="""
        This is the epilog.
        """, arguments=('mice', 'tags', 'dates'))
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
    # arg_parser.add_argument(
    #     "-H", "--hungry_sated", choices=(0, 1, 2), type=int, default=0,
    #     help="0 is hungry trials, 1 is sated trials, 2 is hungry-sated")

    args = arg_parser.parse_args()

    return args


def generate_fig(sorter, trace_type, t_range_s, errortrials, baseline):
    """Do the real plotting."""
    fig, axs = flow.misc.plotting.layout_subplots(
        len(sorter), width=16, height=9, sharey=False, sharex=True)
    for date, ax in zip(sorter, axs.flat):
        pps.stimulus_mean_response(
            ax, date, plot_all=False, trace_type=trace_type,
            start_s=t_range_s[0], end_s=t_range_s[1], errortrials=errortrials,
            baseline=baseline)
    return fig


def main():
    """Main function."""
    filename = 'stimulus_response_{}.pdf'.format(misc.datestamp())
    save_path = os.path.join(
        flow.paths.graphd, 'stimulus_response', filename)
    args = parse_args()
    sorter = metadata.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    fig = generate_fig(
        sorter, args.trace_type, args.t_range_s, args.errortrials,
        args.baseline)
    summary_fig = misc.summary_page(sorter, figsize=(16, 9), **vars(args))
    misc.save_figs(save_path, [summary_fig, fig])
    print(save_path)


if __name__ == '__main__':
    main()
