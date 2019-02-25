"""Look at stimulus responses."""
from __future__ import print_function
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    mpl.use('Agg')

import flow
from flow import misc
import pool
from pool.layouts import reactivation as plr


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Script to plot mean stimulus response over days.""",
        epilog="""
        The 'dates' option really only makes sense with a single Mouse.
        """, arguments=('mice', 'tags', 'dates', 'overwrite', 'verbose'))
    arg_parser.add_argument(
        "-R", "--t_range_s", nargs=2, type=int, default=(-5, 10),
        help="Time range around stimulus to plot. If second value <= first or <=0, include all time up to next stim presentation.")

    args = arg_parser.parse_args()

    return args


def main():
    """Main function."""
    filename = '{}_trial_reactivation.pdf'
    save_dir = os.path.join(flow.paths.graphd, 'trial_reactivation')
    args = parse_args()
    sorter = flow.MouseSorter.frommeta(
        mice=args.mice, tags=args.tags)

    pre_s = -1 * args.t_range_s[0]
    post_s = args.t_range_s[1] if \
        args.t_range_s[1] > args.t_range_s[0] and args.t_range_s[1] > 0 else None

    adb = pool.database.db()

    for mouse in sorter:
        save_path = os.path.join(save_dir, filename.format(mouse))
        if not args.overwrite and os.path.exists(save_path):
            continue

        if args.verbose:
            print('Generating trial reactivation plots: {}'.format(mouse))

        dates = mouse.dates(dates=args.dates, tags=args.tags)
        summary_fig = misc.summary_page(
            dates, figsize=(16, 9), **vars(args))
        figs = [summary_fig]
        for date in dates:
            fig = plr.probability_throughout_trials(
                date.runs(run_types=['training'], tags=args.tags),
                pre_s=pre_s, post_s=post_s)
            dprime = adb.get('behavior_dprime_orig', date.mouse, date.date)
            fig.suptitle('{}\ndprime = {:.2f}'.format(date, dprime))
            figs.append(fig)

        misc.save_figs(save_path, figs)
        print(save_path)

if __name__ == '__main__':
    main()
