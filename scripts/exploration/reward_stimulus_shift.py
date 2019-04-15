import numpy as np

import flow.grapher
import flow.metadata
import flow.misc
import flow.paths
import flow.sorters
import pool.calc.clusters
import pool.database
import pool.plotting.colors as colors

# from flow import misc
# import flow.categories
# import flow.misc.math
# import flow.grapher as grapher
# import flow.netcom as netcom
# import flow.paths
# import pool.config
# import pool.database
# import pool.plotting.colors as colors

def main():
    """Main function."""
    args = parse_args()
    andb = pool.database.db()
    sorter = flow.sorters.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    np.warnings.filterwarnings('ignore')

    gr_rn = flow.grapher.graph(args.save_path, 'half')
    gr_rew = flow.grapher.graph(args.save_path, 'half')
    gr_non = flow.grapher.graph(args.save_path, 'half')

    x = args.stimulus_length/np.arange(args.number_of_splits)

    for d, date in enumerate(sorter):
        nonlbls, rewlbls = pool.calc.clusters.reward(date)
        dist = float(d)/(len(sorter) - 1)
        clr = colors.merge(colors.color('red'),
                           dist,
                           mergeto=colors.hex2tuple(colors.color('blue')))
        clr = colors.tuple2hex(clr)

        if np.sum(nonlbls) > 0 and np.sum(rewlbls) > 0:
            for run in date.runs('training'):
                print(run.mouse, run.date, run.run)
                t2p = run.trace2p()
                # ncells x ntimes x nonsets
                trs = t2p.cstraces('plus', trace_type='deconvolved', start_s=0, end_s=args.stimulus_length)
                nonsets = np.shape(trs)[2]

                split = int(-(-np.shape(trs)[1]//args.number_of_splits))
                non_lbl_mat = np.tile(nonlbls, (nonsets, 1)).transpose()
                rew_lbl_mat = np.tile(rewlbls, (nonsets, 1)).transpose()
                non_counts = np.zeros((nonsets, args.number_of_splits))
                rew_counts = np.zeros((nonsets, args.number_of_splits))

                for i in range(args.number_of_splits):
                    split_trace = np.nanmax(trs[:, i*split:(i+1)*split, :], axis=1)
                    split_trace = (split_trace > args.deconvolved_threshold).astype(np.bool)
                    non_counts[:, i] = np.sum(np.bitwise_and(split_trace, non_lbl_mat), axis=0).astype(np.float64)
                    rew_counts[:, i] = np.sum(np.bitwise_and(split_trace, rew_lbl_mat), axis=0).astype(np.float64)

                trajectories = rew_counts/(rew_counts + non_counts)

                for i in range(nonsets):
                    gr_rn.add(x, trajectories[i, :], **{'opacity': 0.01, 'color':clr})
                    gr_non.add(x, non_counts[i, :], **{'opacity': 0.01, 'color':clr})
                    gr_rew.add(x, rew_counts[i, :], **{'opacity': 0.01, 'color':clr})

    gr_rn.line(**{
        'ymin':0,
        'ymax':1,
        'ytitle':'Rewardiness (1=reward-rich)',
        'xtitle':'Time during stim (0.5 second chunks)',
        'save':'%s stim rewardiness'%sorter.name,
    })

    gr_non.line(**{
        'ymin': 0,
        'ytitle': 'Number of non-reward cells active',
        'xtitle': 'Time during stim (0.5 second chunks)',
        'save': '%s stim non-reward count'%sorter.name,
    })

    gr_rew.line(**{
        'ymin': 0,
        'ytitle': 'Number of reward cells active',
        'xtitle': 'Time during stim (0.5 second chunks)',
        'save': '%s stim reward counts'%sorter.name,
    })


def parse_args():
    arg_parser = flow.misc.default_parser(
        description="""
        Plot clusters across pairs of days.""",
        arguments=('mice', 'tags', 'dates'))
    arg_parser.add_argument(
        '-v', '--visually_driven', type=int, default=50,
        help='The visual-drivenness threshold. Standard is 50.')
    arg_parser.add_argument(
        '-l', '--stimulus_length', type=float, default=2.0,
        help='The length of the stimulus, usually two seconds.')
    arg_parser.add_argument(
        '-n', '--number_of_splits', type=float, default=4,
        help='Number of times in which to split the stimulus.')
    arg_parser.add_argument(
        '-th', '--deconvolved_threshold', type=float, default=0.2,
        help='The deconvolved threshold above which a cell is considered active.')
    arg_parser.add_argument(
        '-p', '--save_path', type=str, default=flow.paths.graphcrossday(),
        help='The directory in which to save the output graph.')

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    main()
