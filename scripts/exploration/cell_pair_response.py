"""Look at stimulus responses."""
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
import os.path as opath

import flow
from flow import misc
import flow.grapher as grapher
import flow.metadata as metadata
import pool.plotting.colors as colors
import pool.database


def heatmap(ax, activity1, outcomes1, activity2, outcomes2, trange):
    """
    Draw a heatmap on an axis.

    Parameters
    ----------
    ax
    activity1
    outcomes1
    activity2
    outcomes2
    trange
    ttype

    Returns
    -------

    """

    act = np.concatenate([activity1, activity2], axis=1).T
    out = np.concatenate([outcomes1, outcomes2])
    nframes = np.shape(act)[1]
    ntrials = np.shape(act)[0]

    im = ax.pcolormesh(np.linspace(trange[0], trange[1], nframes),
                       np.arange(ntrials), act, vmin=-0.25, vmax=0.25, cmap=colors.gradbr())
    im.set_rasterized(True)

    # Plot separation between the two days
    ax.plot([trange[0], trange[1]],
            [np.shape(activity1)[1], np.shape(activity1)[1]],
            color=colors.color('black'), linewidth=2)

    # Plot outcome times
    for i in range(len(out)):
        if out[i] > 0:
            ax.plot([out[i], out[i]], [i, i+1])

    return ax


def smooth(x, window_len=5, window='flat'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError('Smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len/2-1):-(window_len/2)]


def stimuli(runs, cell, stimulus, trange):
    """
    Get the trial responses to each stimulus.

    Parameters
    ----------
    runs : list of Runs
    cell : int
        The position of the cell
    stimulus : str
        The name of the stimulus, usually plus
    trange : tuple
        Time range to display in seconds

    Returns
    -------
    numpy matrix, numpy array
        The activity of ntrials x nframes, the time of each outcome
    """

    # Get the activity and outcomes for each trial
    activity, outcomes = [], []
    for run in runs:
        t2p = run.trace2p()
        act = t2p.cstraces(stimulus, start_s=trange[0], end_s=trange[1],
                           trace_type='dff', baseline=(-1, 0))[cell, :, :]
        outc = t2p.outcomes(stimulus).astype(np.float64)/t2p.framerate

        activity = np.concatenate([activity, act], axis=1) if len(activity) > 0 else act
        outcomes = np.concatenate([outcomes, outc])

    # Order the trials correctly
    timing = np.copy(outcomes)
    timing[timing < 0] = 999
    order = np.argsort(timing)[::-1]
    activity = activity[:, order]
    outcomes = outcomes[order]

    # And smooth
    # for t in range(np.shape(activity)[1]):
    #     smoothed = smooth(activity[:, t])[:len(activity[:, t])]
    #     activity[:len(smoothed), t] = smoothed

    return activity, outcomes


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Plot trial responses of pairs of cells that match criteria.""",
        arguments=('mice', 'tags', 'dates'))
    arg_parser.add_argument(
        '-D', '--day_distance', nargs=2, type=int, default=(0, 6),
        help='Distance between days, inclusive.')
    arg_parser.add_argument(
        '-s', '--sequential', action="store_false",
        help='Limit only to sequential recording days.')
    arg_parser.add_argument(
        '-r', '--cross_reversal', action="store_true",
        help='Allow day pairs across reversal if true.')
    arg_parser.add_argument(
        '-R', '--trange_s', nargs=2, type=int, default=(-1, 6),
        help='Time range around stimulus to plot.')
    arg_parser.add_argument(
        '-l', '--limit', nargs=3, default=('glm-devexp-ensure', '>', '0.02',),
        help='Limitation on cells to display, should be three parts of analysis comparator value')
    arg_parser.add_argument(
        '-v', '--visually_driven', type=int, default=50,
        help='The visual-drivenness threshold. Standard is 50.')

    args = arg_parser.parse_args()

    return args


def main():
    """Main function."""
    filename = '{}_{}_{}_{}_cell_response'
    save_dir = opath.join(flow.paths.graphd, 'cell_response')
    args = parse_args()
    andb = pool.database.db()
    sorter = metadata.DatePairSorter.frommeta(
        mice=args.mice, dates=args.dates, day_distance=args.day_distance, sequential=args.sequential,
        cross_reversal=args.cross_reversal, tags=args.tags)

    for day1, day2 in sorter:
        lims1 = andb.get(args.limit[0], day1.mouse, day1.date)[day1.cells]
        lims2 = andb.get(args.limit[0], day2.mouse, day2.date)[day2.cells]
        binarized = (lims2 > float(args.limit[2]) if args.limit[1] == '>'
            else lims2 >= float(args.limit[2]) if args.limit[1] == '>='
            else lims2 < float(args.limit[2]) if args.limit[1] == '<'
            else lims2 <= float(args.limit[2]))

        if args.visually_driven > 0:
            vdrive1 = andb.get('visually-driven-plus', day1.mouse, day1.date)[day1.cells]
            vdrive2 = andb.get('visually-driven-plus', day2.mouse, day2.date)[day2.cells]
            vdrive = np.bitwise_and(vdrive1 > args.visually_driven, vdrive2 > args.visually_driven)
            binarized = np.bitwise_and(binarized, vdrive)

        for cell in np.arange(len(binarized))[binarized]:
            activity1, outcomes1 = stimuli(day1.runs('training'), cell, 'plus', args.trange_s)
            activity2, outcomes2 = stimuli(day2.runs('training'), cell, 'plus', args.trange_s)

            gr = grapher.graph(save_dir, 'full')
            print cell
            import pdb;pdb.set_trace()
            ax = heatmap(gr.axis(), activity1, outcomes1, activity2, outcomes2, args.trange_s)
            gr.graph(ax, **{
                # 'xmin': args.trange_s[0],
                # 'xmax': args.trange_s[1],
                'save': filename.format(day1.mouse, day1.date, day2.date, cell),
                'title': '%s %.2f->%.2f' % (args.limit[0], lims1[cell], lims2[cell])
            })


if __name__ == '__main__':
    main()
