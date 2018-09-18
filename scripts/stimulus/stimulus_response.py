"""Look at stimulus responses."""
import matplotlib.pyplot as plt
import os
from sys import argv

from pool.plotting import stimulus as pps
from pool.layouts import stimulus as pls

import flow
from flow import misc
from flow.metadata2 import DateSorter


def parse_args_orig():
    """Parse arguments."""
    defaults = {
        'graph': 'stimulus',
        'sort': '',  # Analysis upon which to sort
        'analyses': [],  # Analyses to display along right side
        'color-max': 'auto',
        'trace-type': 'dff',
        'trange-ms': (-1000, 8000),
        'baseline-ms': (-1000, 0),
        'display': 'dff',
        'remove-licking': False,
        'error-trials': -1,  # -1 is off, 0 is correct trials, 1 is error trials, 2 is diff of error trials
        'hungry-sated': 0,  # 0 is hungry trials, 1 is sated trials, 2 is hungry-sated
    }

    lpars = parseargv.extractkv(argv, defaults)
    if not isinstance(lpars['analyses'], list):
        lpars['analyses'] = [lpars['analyses']]

    days = parseargv.sorteddays(argv, classifier=False, trace=False, force=True)

    print lpars, days
    from pudb import set_trace; set_trace()


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


def save_pdf(figs, save_path):
    """Save figs to a pdf."""
    # header_fig = misc.summary_page(days, lpars)
    # figs.insert(0, header_fig)
    misc.save_figs(save_path, figs)


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


def traces_per_cell(sorter, trace_type, t_range_s, errortrials, baseline):
    for date in sorter:
        for roi_idx in [0,1,2]:
            fig = plt.figure()
            pls.all_stimulus_traces(
                fig, date, roi_idx, start_s=t_range_s[0], end_s=t_range_s[1],
                trace_type=trace_type, errortrials=errortrials,
                baseline=baseline)
            from pudb import set_trace; set_trace()



def main():
    """Main function."""
    filename = 'stimulus_response_{}.pdf'.format(misc.datestamp())
    save_path = os.path.join(
        flow.paths.graphd, 'stimulus_response', filename)
    args = parse_args()
    sorter = DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)

    traces_per_cell(sorter, args.trace_type, args.t_range_s, args.errortrials,
                    args.baseline)


    fig = generate_fig(
        sorter, args.trace_type, args.t_range_s, args.error_trials,
        args.baseline)
    save_pdf([fig], save_path)
    print(save_path)


if __name__ == '__main__':
    main()
    # parse_args()
