"""Look at stimulus responses."""
from __future__ import division, print_function
from builtins import range

import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
import os.path as opath

import flow
from flow import misc
from flow import sorters
import flow.misc.math
import flow.grapher as grapher
import pool.config
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


def trace(activity, save_dir, filename, trange):
    """

    Parameters
    ----------
    activity
    save_dir
    filename
    trange

    Returns
    -------

    """

    gr = grapher.graph(save_dir, 'half')
    tau1 = 0.8670

    colors = ['mint', 'indigo']
    for i, act in enumerate(activity):
        x = np.linspace(trange[0], trange[1], np.shape(act)[0])
        mu = np.nanmean(act, axis=1)
        error = np.nanstd(act, axis=1)/np.sqrt(np.shape(act)[1])
        gr.add(x, mu, **{'color': colors[i], 'errors': error})

        pos = np.argmax(x >= 2) - 1
        subx = np.linspace(0, trange[1] - 2, len(x[pos+1:])) + (x[pos+1] - 2)
        decay = np.zeros(len(mu))
        decay[-len(subx):] = mu[pos]*np.exp(-subx/tau1)
        gr.add(x, mu - decay, **{'color': 'orange', 'errors': error})

    gr.line(**{
        'xtitle': 'TIME (s)',
        'ytitle': 'DFF',
        'xmin': trange[0],
        'xmax': trange[1],
        'ymin': -0.25,
        'ymax': 0.25,
        'save': filename,
    })


def stimuli(runs, cell, stimulus, trange, trace_type='dff', error_trials=-1, smooth=True):
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
    trace_type : str
        Trace type (e.g. 'dff', 'zscore')
    error_trials : int
        -1 all trials, 0 correct trials, 1 error trials
    smooth : bool
        If true, smooth the activity

    Returns
    -------
    numpy matrix, numpy array
        The activity of ntrials x nframes, the time of each outcome
    """

    # Get the activity and outcomes for each trial
    activity, outcomes = [], []
    baseline = None if trace_type == 'deconvolved' else (-1, 0)
    for run in runs:
        t2p = run.trace2p()
        act = t2p.cstraces(stimulus, start_s=trange[0], end_s=trange[1], trace_type=trace_type,
                           errortrials=error_trials, baseline=baseline)[cell, :, :]
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
    if smooth:
        for t in range(np.shape(activity)[1]):
            smoothed = flow.misc.math.smooth(activity[:, t], window_len=3)[:len(activity[:, t])]
            activity[:len(smoothed), t] = smoothed

    return activity, outcomes


def glm_activity(runs, cell, stimulus, trange):
    """
    Return the reconstructed GLM activity based on the number of stimulus presentations.

    Parameters
    ----------
    runs : RunSorter
    cell : int
        Cell number
    stimulus : str
        Event type to return GLM from
    trange : tuple of ints
        Time range in seconds

    Returns
    -------
    matrix of size (nframes, 1)
        Response of the cell

    """

    model = runs[0].parent().glm()
    if model is None or not model:
        raise ValueError('No GLM found for %s %i', runs[0].mouse, runs[0].date)

    responses = model.meanresp(trange, hz=15.49)

    out, total = None, 0
    if stimulus in pool.config.stimuli():
        for run in runs:
            t2p = run.trace2p()
            ncorrect = len(t2p.csonsets(stimulus, errortrials=0))
            nerror = len(t2p.csonsets(stimulus, errortrials=1))

            if stimulus == 'plus':
                ncorrect += len(t2p.csonsets('pavlovian', errortrials=-1))

            if out is None:
                out = np.zeros(len(responses['%s_correct' % stimulus][cell, :]))

            out += ncorrect*responses['%s_correct' % stimulus][cell, :]
            out += nerror*responses['%s_miss' % stimulus][cell, :]
            total += ncorrect + nerror

        out /= float(total)
    else:
        out = responses[stimulus][cell, :]

    return out.reshape((len(out), 1))


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
        '-R', '--trange_s', nargs=2, type=int, default=(-2, 4),
        help='Time range around stimulus to plot.')
    arg_parser.add_argument(
        '-l', '--limit', nargs=3, default=('devexp_ensure', '>', '0.02',),
        help='Limitation on cells to display, should be three parts of analysis comparator value')
    arg_parser.add_argument(
        '-v', '--visually_driven', type=int, default=50,
        help='The visual-drivenness threshold. Standard is 50.')

    args = arg_parser.parse_args()

    return args


def main():
    """Main function."""
    filename = '{}_{}_{}_{}_cell_response{}'
    save_dir = opath.join(flow.paths.graphd, 'cell_response')
    args = parse_args()
    andb = pool.database.db()
    sorter = sorters.DatePairSorter.frommeta(
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
            vdrive1 = andb.get('vdrive_plus', day1.mouse, day1.date)[day1.cells]
            vdrive2 = andb.get('vdrive_plus', day2.mouse, day2.date)[day2.cells]
            vdrive = np.bitwise_and(vdrive1 > args.visually_driven, vdrive2 > args.visually_driven)
            binarized = np.bitwise_and(binarized, vdrive)

        for cell in np.arange(len(binarized))[binarized]:
            activity1, outcomes1 = stimuli(day1.runs('training'), cell, 'plus', args.trange_s)
            activity2, outcomes2 = stimuli(day2.runs('training'), cell, 'plus', args.trange_s)

            gr = grapher.graph(save_dir, 'full')
            print(cell)
            import pdb;pdb.set_trace()
            ax = heatmap(gr.axis(), activity1, outcomes1, activity2, outcomes2, args.trange_s)
            gr.graph(ax, **{
                # 'xmin': args.trange_s[0],
                # 'xmax': args.trange_s[1],
                'save': filename.format(day1.mouse, day1.date, day2.date, cell),
                'title': '%s %.2f->%.2f' % (args.limit[0], lims1[cell], lims2[cell])
            })

            plus, _ = stimuli(day1.runs('training'), cell, 'plus', args.trange_s, 'dff')
            reward, _ = stimuli(day1.runs('training'), cell, 'ensure', args.trange_s, 'dff')
            trace([plus, reward], save_dir,
                  filename.format(day1.mouse, day1.date, day2.date, cell, '_dff_day1'), args.trange_s)

            x = np.linspace(args.trange_s[0], args.trange_s[1], np.shape(plus)[0])

            plus, _ = stimuli(day2.runs('training'), cell, 'plus', args.trange_s, 'dff')
            reward, _ = stimuli(day2.runs('training'), cell, 'ensure', args.trange_s, 'dff')
            trace([plus, reward], save_dir,
                  filename.format(day2.mouse, day1.date, day2.date, cell, '_dff_day2'), args.trange_s)

            print(np.mean(plus[(x >= 0) & (x <= 2)]))
            print(np.mean(plus[(x >= 2) & (x <= 4)]))
            print(np.mean(reward[(x >= 0) & (x <= 2)]))


if __name__ == '__main__':
    main()
