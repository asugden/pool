import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np

from flow import paths
import flow.metadata as metadata

from pool import config
from pool import stimulusdff
from pool.plotting import graphfns


def axheatmaplabels(fig, ax, data, framerate):
    """
    Add stimulus-specific labels to a figure axis
    :param fig: mpl figure
    :param ax: mpl axis
    :return:
    """

    ncells = np.shape(data)[0]
    nframes = np.shape(data)[1]
    clrs = config.colors()

    xticks = []
    xticklabels = []
    for i in range(3):
        xticks.extend([framerate + i*framerate*9, framerate*3 + i*framerate*9])
        xticklabels.extend(['0', '2'])
        ax.plot([i*nframes/3., i*nframes/3.], [0, ncells], lw=1.5, color='#000000')

    ax.text(nframes/6, ncells*1.01, 'PLUS STIMS', {'color': clrs['plus'], 'ha': 'center', 'size': 14})
    ax.text(3*nframes/6, ncells*1.01, 'NEUTRAL STIMS', {'color': clrs['neutral'], 'ha': 'center', 'size': 14})
    ax.text(5*nframes/6, ncells*1.01, 'MINUS STIMS', {'color': clrs['minus'], 'ha': 'center', 'size': 14})

    ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xlim((0, nframes))
    ax.set_ylim((0, ncells))

def horgraph(fig, ax, vals, borders, xticks, title, clr):
    """
    Add a horizontal chart of the marginal probabilities
    :param fig:
    :param ax:
    :param vals:
    :return:
    """

    # settings.colors()[cs]
    ncells = len(vals)

    plt.barh(np.arange(ncells) + 0.5, vals, color=clr, edgecolor=None, linewidth=0)

    #ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(xticks)
    # ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # ax.axvline(0, 0.0, len(marginals), linewidth=4, color=colors.color('gray'))

    # ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axvline(0, lw=1.5, color='#7C7C7C', clip_on=False)
    ax.text((xticks[-1] - xticks[0])/2.0 + xticks[0], len(vals)*1.01, title, {'ha': 'center', 'size': 14})

    ax.set_xlim((xticks[0], xticks[-1]))
    ax.set_ylim((0, len(vals)))

    clrs = config.colors()
    for gr in borders:
        ax.plot([xticks[0], xticks[-1]], [ncells - borders[gr], ncells - borders[gr]], lw=1.5, color=clrs[gr], clip_on=False)

def sortorder(andb, mouse, date, analysis=''):
    """
    Return the sort order based on an analysis and date
    :param date:
    :param analysis:
    :return: sort order
    """

    borders = {}
    if analysis == '':
        return andb.get('sort_order', mouse, date), \
            andb.get('sort_borders', mouse, date)

    vals = andb.get(analysis, mouse, date)
    order = sorted([(v, i) for i, v in enumerate(vals)])
    return np.array([i[1] for i in order]), {}

def heatmaptype(fig, ax, args, lpars, sorting, borders):
    """
    Plot the correct type of heatmap
    :param args:
    :param lpars:
    :return:
    """

    tlpars = {
        'trange-ms': (-1000, 8000),
        'baseline-ms': lpars['baseline-ms'],
        'trace-type': lpars['trace-type'],
        'cutoff-before-lick-ms': 100 if lpars['remove-licking'] else -1,
        'error-trials': lpars['error-trials'],
        'ordered-cses': ['plus', 'neutral', 'minus'],
    }
    if lpars['error-trials'] == 2: tlpars['error-trials'] = 0
    if 'add-ensure-quinine' in args[-1] and args[-1]['add-ensure-quinine']:
        tlpars['ordered-cses'] += ['ensure', 'quinine']

    md = metadata.data(args[-1]['mouse'], args[-1]['training-date'])
    if 'sated-stim' not in md: md['sated-stim'] = []
    runs = md['training']
    if lpars['hungry-sated'] == 1: runs = md['sated-stim'] + md['sated']
    elif lpars['hungry-sated'] < 0: runs += md['sated-stim'] + md['sated']

    plot, framerate = stimulusdff.dff(args[-1], tlpars, runs)

    if lpars['error-trials'] == 2 or lpars['hungry-sated'] == 2:
        runs = md['training']
        if lpars['hungry-sated'] == 1: runs = md['sated-stim'] + md['sated']
        elif lpars['hungry-sated'] < 0: runs += md['sated-stim'] + md['sated']
        if lpars['hungry-sated'] == 2: runs = md['sated-stim'] + md['sated']
        if lpars['error-trials'] == 2: tlpars['error-trials'] = 1

        plot2, _ = stimulusdff.dff(args[-1], tlpars, runs)
        plot -= plot2

    plot = plot[sorting, :]
    graphfns.axheatmap(fig, ax, plot, borders, lpars['trace-type'], lpars['color-max'])
    axheatmaplabels(fig, ax, plot, framerate)
    graphfns.axytitle(fig, ax, '%i CELLS' % (np.shape(plot)[0]))

def corrgraphscale(name, vals):
    """
    Set the correlation graph scale correctly
    :return:
    """

    if np.min(vals) >= 0 and np.min(vals) <= 1:
        xticks = [0, 0.5, 1]
    elif np.min(vals) >= 0 and np.min(vals) <= 0.5:
        xticks = [0, 0.5]

    if 'chprob' in name or 'auroc' in name:
        vals -= 0.5
        xticks = [-0.5, 0, 0.5]

    if 'rwa' in name:
        # vals *= 5000
        xticks = [0, 1]
    if 'rwr' in name:
        xticks = [-0.3, 0.3]
    if 'ff' in name or 'fano' in name:
        xticks = [0, 1, 2]
    if 'mutual-information' in name:
        xticks = [0, 0.5]
    if 'hunger-modulation' in name:
        xticks = [-10, 0, 10]

    return vals, xticks

def setgraph(args, andb, mouse, lpars):
    """
    Set and make graph, accounting for width
    :param args: general parameters, from settings
    :param andb: analysis database
    :param lpars: local parameters
    :return: None
    """

    # Set the widths
    widths = []
    if lpars['graph'].lower() != 'none': widths.append(6)
    for an in lpars['analyses']: widths.append(2)

    w = 14.35*0.77
    if np.sum(widths) > 12: w = 1.2*np.sum(widths)

    # Star the figure
    fig = plt.figure(figsize=(w, 6))
    gs = grd.GridSpec(1, len(widths), width_ratios=widths)
    graphfns.style(sz=16)

    # Get the sorting order
    sorting, borders = graphfns.sortorder(andb, mouse, args[-1]['training-date'], lpars['sort'])

    fign = 0
    if lpars['graph'].lower() != 'none':
        ax = plt.subplot(gs[fign])
        heatmaptype(fig, ax, args, lpars, sorting, borders)
        fign += 1

    for an in lpars['analyses']:
        ax = plt.subplot(gs[fign])
        vals = andb.get(an, mouse, args[-1]['training-date'])
        if vals is None or (isinstance(vals, float) and np.isnan(vals)) or len(vals) == 0: return
        vals, xscale = corrgraphscale(an, vals[sorting])
        clr = 'plus' if 'plus' in an else 'neutral' if 'neutral' in an else 'minus' if 'minus' in an else 'other'
        horgraph(fig, ax, vals, borders, xscale, an.replace('-', ' ').upper(), config.colors()[clr])
        fign += 1

    # Get the graph base path with mouse, date, and run prepended
    path = paths.graphmdr(args[-1])
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    graphfns.axtitle(fig, ax, '%s-%s' % (args[-1]['mouse'], args[-1]['comparison-date']))
    plt.subplots_adjust(top=0.95)

    savevals = [lpars['graph'], '-'.join(lpars['analyses'])]
    path = '%s-%s'%(path, '-'.join(k for k in savevals if len(k) > 0))
    print path + '.pdf'
    graphfns.save(fig, path)


if __name__ == '__main__':
    from sys import argv
    from flow import parseargv
    from pool import database

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
    if not isinstance(lpars['analyses'], list): lpars['analyses'] = [lpars['analyses']]
    first = True
    days = parseargv.sorteddays(argv, classifier=False, trace=False, force=True)
    while days.next():
        md, args = days.get()
        if first:
            first = False
            andb = database.db()

        setgraph(args, andb, md[0], lpars)

    days.reset()
    andb.save()
