"""Look at stimulus responses."""
import matplotlib.pyplot as plt
import os
from sys import argv

from pool.plotting import stimulus_plotting as sp

import flow
from flow.misc import misc
from flow.classes import DateSorter


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
    defaults = {
        'graph': 'stimulus',
        'sort': '',  # Analysis upon which to sort
        # 'analyses': [],  # Analyses to display along right side
        'color-max': 'auto',
        'trace-type': 'dff',
        # 'trange-ms': (-1000, 8000),
        # 'baseline-ms': (-1000, 0),
        'display': 'dff',
        'remove-licking': False,
        'error-trials': -1,  # -1 is off, 0 is correct trials, 1 is error trials, 2 is diff of error trials
        'hungry-sated': 0,  # 0 is hungry trials, 1 is sated trials, 2 is hungry-sated
    }

    arg_parser = misc.smart_parser()
    arg_parser.add_argument(
        "-m", "--mouse", action="store",
        help="Mice to analyze")
    arg_parser.add_argument(
        "-t", "--trace_type", action="store", default="dff",
        help="Trace type to plot.")

    args = arg_parser.parse_args()

    lpars = flow.parseargv.parseargs(args, defaults)

    return lpars


def save_pdf(figs, save_path, days, lpars):
    """Save figs to a pdf."""
    # header_fig = misc.summary_page(days, lpars)
    # figs.insert(0, header_fig)
    misc.save_figs(save_path, figs)


def generate_fig(sorter, lpars):
    """Do the real plotting."""
    fig, axs = misc.layout_subplots(
        len(sorter), width=16, height=9, sharey=True, sharex=True)
    for date, ax in zip(sorter, axs.flat):
        sp.stimulus_mean_response(ax, date, plot_all=False, trace_type=lpars['trace_type'])
    return fig


def main():
    """Main function."""
    filename = 'stimulus_response_{}.pdf'.format(misc.datestamp())
    save_path = os.path.join(
        flow.paths.graphd, 'stimulus_response', filename)
    lpars = parse_args()
    # sorter = sorters.RunSorter.frommice(
    #     [lpars['mouse']], spontaneous=True, training=True)
    sorter = DateSorter.frommice(
        [lpars['mouse']])
    fig = generate_fig(sorter, lpars)
    save_pdf([fig], save_path, sorter, lpars)
    print(save_path)


if __name__ == '__main__':
    main()
    # parse_args()
    # parse_args_orig()
