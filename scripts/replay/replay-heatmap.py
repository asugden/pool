from __future__ import division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object, range, str
from subprocess import getoutput

from copy import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
import os.path as opath
try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

import flow
from flow import classify2p, config, misc, paths

import pool
from pool.plotting import colors, graphfns


class ClassifierHeatmap(object):
    def __init__(self, path, ts, trs, cls={}, beh={}, bord={}, top={}, clrs={}, zoom=-1, tracetype='deconvolved'):
        """
        Instantiates a Classifier Graph. Takes pars information directly
        from ClassifyAODE, as well as a trace2p trace and the output
        from the classifier.
        """

        nframes = np.shape(trs)[1]
        fig, gs = self.getfig(nframes, top!={}, zoom)
        graphfns.style(sz=20)

        if top != {}:
            self.toptrace(plt.subplot(gs[0]), ts, top)

        self.behavior(plt.subplot(gs[-3]), ts, beh)
        self.classifier(plt.subplot(gs[-2]), ts, cls, clrs)
        self.heatmap(plt.subplot(gs[-1]), ts, trs, bord, clrs, tracetype, beh)

        plt.tight_layout()
        with graphfns.SuppressErrors():
            plt.savefig(path, transparent=False, dpi=200)
        plt.close(fig)
        plt.close('all')

        plt.clf()
        plt.cla()
        plt.gcf().clear()

    def getfig(self, nframes, graphtop=True, zoom=-1):
        """
        Get the figure and gridspec obections
        :param nframes: number of frames
        :param graphtop: True if the top graph should be included, else False
        :return: fig, gridspec
        """

        # Set the width of the image based on number of frames presented
        if zoom < 0:
            if nframes > 5000:
                zoom = 0
            elif nframes > 1000:
                zoom = 1
            else:
                zoom = 2

        framescale = {0: 0.002, 1: 0.0054, 2: 0.01}
        width = max(10, framescale[zoom]*nframes)

        height = 8.5 if not graphtop else 10
        hrs = [1.5, 2, 8] if not graphtop else [1.5, 1.5, 2, 8]

        fig = plt.figure(figsize=(width, height))
        gs = grd.GridSpec(len(hrs), 1, height_ratios=hrs)

        return fig, gs

    def toptrace(self, ax, ts, top):
        """
        Plot the running, brain motion, and pupil diameter at the top of
        the plot.
        """

        # 0-linewidth, 1-color, 2-opacity, 3-yrange, 4-display axis
        vstyle = {
            'photometry': [1, colors.color('indigo'), 1, (0, 1.001), 0],
            'ripple': [1, colors.color('gray'), 1, (0, 1), 0],
            'pop-activity': [1, colors.color('black'), 1, (0, 1.001), 0],
        }
        # Set the order for naming importance
        order = ['ripple', 'photometry', 'pop-activity']

        # Set the visual style of the axis
        ax = self._simpleaxis(ax)

        for key in order:
            if key in vstyle and key in top:
                if vstyle[key][4] > 0:
                    tx = ax.twinx()
                    tx = self._simpleaxis(tx)
                else:
                    tx = ax

                x, y = graphfns.reducepoints(ts, top[key], 30000)
                tx.plot(x, y, linewidth=vstyle[key][0], color=vstyle[key][1],
                        alpha=vstyle[key][2])
                tx.set_ylim(vstyle[key][3])
                tx.yaxis.set_ticks(vstyle[key][3])
                tx.set_ylabel(key.upper()[:4], color=vstyle[key][1])
                tx.set_xlim([np.min(ts), np.max(ts)])
                tx.get_xaxis().set_ticklabels([])

    def behavior(self, ax, ts, beh):
        """
        Plot the running, brain motion, and pupil diameter at the top of
        the plot.
        """

        # 0-linewidth, 1-color, 2-opacity, 3-yrange, 4-display axis
        vstyle = {
            'motion': [0.5, colors.color('red'), 0.5, (-2, 2), 1],
            'pupil': [1, colors.color('gray'), 1, (0, 1), 0],
            'pupil-mask': [3, colors.color('orange'), 1, (0, 1), 0],
            'running': [1, colors.color('blue'), 0.9, (0, 1), 0],
        }
        # Set the order for naming importance
        order = ['pupil', 'pupil-mask', 'motion', 'running']

        # Set the visual style of the axis
        ax = self._simpleaxis(ax)

        for key in order[::-1]:
            if key in vstyle and key in beh:
                if vstyle[key][4] > 0:
                    tx = ax.twinx()
                    tx = self._simpleaxis(tx)
                else:
                    tx = ax

                # if np.sum(np.invert(np.isfinite(beh[key]))) > 100:
                # 	x, y = np.copy(ts), np.copy(beh[key])
                # 	x, y = x[np.isfinite(y)], y[np.isfinite(y)]

                x, y = graphfns.reducepoints(ts, beh[key])
                tx.plot(x, y, linewidth=vstyle[key][0], color=vstyle[key][1],
                        alpha=vstyle[key][2])
                tx.set_ylim(vstyle[key][3])
                tx.yaxis.set_ticks(vstyle[key][3])
                tx.set_ylabel(key.upper()[:3], color=vstyle[key][1])
                tx.set_xlim([np.min(ts), np.max(ts)])
                tx.get_xaxis().set_ticklabels([])

        # Add licking below the graph
        if 'licking' in beh:
            for lick in beh['licking']:
                ax.axvspan(
                    lick, lick + 1.0/60.0, ymin=-0.05, ymax=-0.15,
                    clip_on=False, zorder=100, color=colors.color('purple'),
                    lw=0, alpha=0.6)

    def classifier(self, ax, ts, cls, clrs):
        """
        Add a classifier graph to an axis.
        """

        # Set the visual style of the axis
        ax = self._simpleaxis(ax)

        # Loop through all saved data and plot
        for cs in cls:
            if cs in clrs:
                x, y = graphfns.reducepoints(ts, cls[cs])
                ax.plot(x, y, linewidth=1, color=clrs[cs])

        ax.set_xlim([np.min(ts), np.max(ts)])
        ax.get_xaxis().set_ticklabels([])

        ax.set_ylim([0, 1])
        ax.yaxis.set_ticks([0, 0.5, 1.0])
        ax.yaxis.set_ticklabels(['0', '', '1.0'])
        ax.set_ylabel('P(REP)')
        # ax.get_yaxis().set_label_coords(-0.1, 0.5)

    def stimuli(self, ax, stims, clrs, correct=True):
        """
        Place vertical bars on the graph to mark stim onsets.
        """

        for cs in stims:
            for ons in stims[cs]:
                ax.axvspan(
                    ons, ons+2.0/60.0, ymin=1.01, ymax=1.03, clip_on=False,
                    zorder=100, color=clrs[cs], lw=0, alpha=0.9)

                if not correct:
                    ax.axvspan(
                        ons, ons+2.0/60.0, ymin=1.01, ymax=1.03, clip_on=False,
                        zorder=100, color='#000000', lw=0, alpha=0.4)

    def heatmap(self, ax, ts, trs, bord, clrs, ttype, beh):
        """
        Add a heatmap to the axis of traces.
        :param ax: axis from gridspec
        :param ts: times, vector
        :param trs: matrix of traces, ncells by ntimes
        :param bord: horizontal borders to draw
        :return:
        """

        if ttype == 'dff':
            vrange = [-0.17, 0.23]
            cmap = colors.gradbr()
        else:
            vrange = [0, 0.1]
            cmap = plt.get_cmap('Greys')

        cells = np.arange(np.shape(trs)[0]) + 1
        im = ax.pcolormesh(ts, cells, trs, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        im.set_rasterized(True)

        trange = [np.min(ts), np.max(ts)]
        for gr in bord:
            grname = gr.split('-')[0]
            ax.plot(trange, [len(cells)-bord[gr], len(cells)-bord[gr]], lw=1.5, color=clrs[grname])

        ax.set_xlim(trange)
        ax.set_ylim([0, len(cells)])
        ax.set_ylabel('%i CELLS'%len(cells))
        ax.yaxis.set_ticklabels([])
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('TIME (m)')

        if 'correct-stimuli' in beh:
            self.stimuli(ax, beh['correct-stimuli'], clrs, True)
        if 'error-stimuli' in beh:
            self.stimuli(ax, beh['error-stimuli'], clrs, False)

            # =====================================================================
            # GRAPHING ESSENTIALS

    def _noaxis(self, ax):
        """
        Set axis to just a scale bar.
        """
        # Ref: https://gist.github.com/dmeliza/3251476

        ax.axis('off')
        return ax

    # Set axis to look like my axes from Keynote into Illustrator
    def _simpleaxis(self, ax):
        ax.yaxis.grid(True, linestyle='solid', linewidth=0.75, color='#AAAAAA')
        ax.tick_params(axis='x', pad=15)
        ax.tick_params(axis='y', pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_ticks_position('none')
        ax.set_axisbelow(True)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        return ax


class ReplayGraphInputs(object):
    """
    Create the inputs that go into a replay graph. A class rather than separate function because time ranges are reused.
    """
    def __init__(self, andb, pars, lpars, classifier, t2p, mouse, date, searchms, plusprob):
        """
        Initialize with all necessary inputs
        :param andb: analysis database, class
        :param lpars: local parameters, dict
        :param pars: parameters that describe the classifier
        :param metadata: metadata from run sorter
        :param classifier: classifier dict
        :param t2p: trace2p instance, class
        :return: None
        """

        if lpars['display-training']:
            self._fmin = 0
        else:
            self._fmin = t2p.lastonset()

        self._trangem = lpars['trange-m']
        self._frange = [int(round(lpars['trange-m'][0]*60.0*t2p.framerate)),
                        int(round(lpars['trange-m'][1]*60.0*t2p.framerate)) + 1]

        self.tr = []
        self._dborders = {}
        self._behtraces = {}
        self._toptraces = {}
        self._clstraces = {}

        self._gettraces(andb, pars, lpars, classifier, t2p, mouse, date)
        self._getbehavior(t2p, lpars)
        self._gettoptrace(lpars, t2p, classifier, plusprob)
        self._getclassifier(lpars, classifier, searchms, t2p)

    def pull_traces(self, frame=-1):
        """
        Pull the traces matrix for all viewable time (if frame < 0) or a particular frame
        :param frame: frame number, int. frame < 0 means all time
        :return: matrix of ncells, ntimes
        """

        if frame < 0:
            return np.copy(self.tr[:, self._fmin:self._maxframe])
        else:
            frange, pad = self._getfrange(frame)
            out = np.copy(self.tr[:, frange[0]:frange[1]])
            if pad[0] > 0:
                nanpad = np.zeros((np.shape(out)[0], pad[0]))
                nanpad[:, :] = np.nan
                out = np.concatenate([nanpad, out], axis=1)
            if pad[1] > 0:
                nanpad = np.zeros((np.shape(out)[0], pad[1]))
                nanpad[:, :] = np.nan
                out = np.concatenate([out, nanpad], axis=1)
            return out

    def pull_behavior(self, frame=-1):
        """
        Pull the behavior dictionary for all viewable time (if frame < 0) or a particular frame
        :param frame:
        :return: dict of behaviors of correct length
        """

        out = {}
        if frame < 0:
            for key in self._behtraces:
                if key == 'licking':
                    vec = np.copy(self._behtraces[key])
                    vec = vec[(vec >= self._fmin) & (vec < self._maxframe)]
                    if len(vec) > 0:
                        out[key] = (vec.astype(np.float32)/
                                    self._maxframe)*self._maxtimem
                elif 'stimuli' in key:
                    outstim = {}
                    for cs in self._behtraces[key]:
                        vec = np.copy(self._behtraces[key][cs])
                        vec = vec[(vec >= self._fmin) & (vec < self._maxframe)]
                        if len(vec) > 0:
                            outstim[cs] = (vec.astype(np.float32)/
                                           self._maxframe)*self._maxtimem
                    if outstim != {}:
                        out[key] = outstim

                else:
                    out[key] = np.copy(
                        self._behtraces[key][self._fmin:self._maxframe])
        else:
            frange, pad = self._getfrange(frame)
            for key in self._behtraces:
                if key == 'licking':
                    vec = np.copy(self._behtraces[key])
                    vec = vec[(vec >= frange[0]) & (vec < frange[1])]
                    if len(vec) > 0:
                        out[key] = (vec.astype(np.float32)/self._maxframe)*self._maxtimem
                elif 'stimuli' in key:
                    outstim = {}
                    for cs in self._behtraces[key]:
                        vec = np.copy(self._behtraces[key][cs])
                        vec = vec[(vec >= frange[0]) & (vec < frange[1])]
                        if len(vec) > 0:
                            outstim[cs] = (vec.astype(np.float32)/self._maxframe)*self._maxtimem
                    if outstim != {}:
                        out[key] = outstim
                else:
                    out[key] = np.copy(self._behtraces[key][frange[0]:frange[1]])
                    if pad[0] > 0:
                        nanpad = np.zeros(pad[0])
                        nanpad[:, :] = np.nan
                        out[key] = np.concatenate([nanpad, out[key]])
                    if pad[1] > 0:
                        nanpad = np.zeros(pad[1])
                        nanpad[:, :] = np.nan
                        out[key] = np.concatenate([out[key], nanpad])

        return out

    def pull_toptrace(self, frame=-1):
        """
        Pull the behavior dictionary for all viewable time (if frame < 0) or a particular frame
        :param frame:
        :return: dict of behaviors of correct length
        """

        out = {}
        if frame < 0:
            for key in self._toptraces:
                out[key] = np.copy(self._toptraces[key][self._fmin:self._maxframe])
        else:
            frange, pad = self._getfrange(frame)
            for key in self._toptraces:
                out[key] = np.copy(self._toptraces[key][frange[0]:frange[1]])
                if pad[0] > 0:
                    nanpad = np.zeros(pad[0])
                    nanpad[:, :] = np.nan
                    out[key] = np.concatenate([nanpad, out[key]])
                if pad[1] > 0:
                    nanpad = np.zeros(pad[1])
                    nanpad[:, :] = np.nan
                    out[key] = np.concatenate([out[key], nanpad])
        return out

    def pull_classifier(self, frame=-1):
        """
        Pull the behavior dictionary for all viewable time (if frame < 0) or a particular frame
        :param frame:
        :return: dict of behaviors of correct length
        """

        out = {}
        if frame < 0:
            for key in self._clstraces:
                out[key] = np.copy(self._clstraces[key][self._fmin:self._maxframe])
        else:
            frange, pad = self._getfrange(frame)
            for key in self._clstraces:
                out[key] = np.copy(self._clstraces[key][frange[0]:frange[1]])
                if pad[0] > 0:
                    nanpad = np.zeros(pad[0])
                    nanpad[:, :] = np.nan
                    out[key] = np.concatenate([nanpad, out[key]])
                if pad[1] > 0:
                    nanpad = np.zeros(pad[1])
                    nanpad[:, :] = np.nan
                    out[key] = np.concatenate([out[key], nanpad])
        return out

    def pull_time(self, frame=-1, unit='m'):
        """
        Pull the the vector of times to be displayed
        :param frame:
        :return: vector of x values in minutes
        """

        if frame < 0:
            fnum = self._maxframe - self._fmin
            tmin = (self._maxtimem/self._maxframe)*self._fmin
            out = np.arange(fnum, dtype=np.float32)/fnum
            out = out*(self._maxtimem - self._tmin) + self._tmin
        else:
            out = np.arange(
                self._frange[1] - self._frange[0],
                dtype=np.float32)/(self._frange[1] - self._frange[0])
            out = out*(self._trangem[1] - self._trangem[0]) + self._trangem[0]

        if unit[0] == 's':
            out *= 60.0
        return out

    def pull_borders(self):
        """
        Pull the borders to be drawn on the graph.
        :return: a dictionary of borders with names that refer to colors
        """

        return self._dborders

    def _getfrange(self, frame):
        """
        Return the frame range to display and the amount to pad with nans
        :return: frange tuple and pad tuple
        """

        fmin = frame + self._frange[0]
        fmax = frame + self._frange[1]

        plo, phi = 0, 0

        if fmin < 0:
            plo = -fmin
            fmin = 0
        if fmax > self._maxframe - 1:
            phi = fmax - (self._maxframe - 1)
            fmax = self._maxframe - 1

        return (fmin, fmax), (plo, phi)

    def _gettraces(self, andb, pars, lpars, classifier, t2p, mouse, date):
        """
        Get the correct type of traces
        :return: None
        """

        boxcar = 6

        # Get the sorting order for display
        sorting, self._dborders = graphfns.sortorder(
            andb, mouse, date, lpars['sort'])
        if 'traces' in classifier:
            self.tr = np.copy(
                classifier['traces'][sorting, :]).astype(np.float64)
        else:
            self.tr = np.copy(t2p.trace(
                lpars['display-type'])[sorting, :]).astype(np.float64)

        if lpars['temporal-filter']:
            actmn, actvar, actouts = classify.activity(pars)
            tprior = classify.temporal_prior(
                self.tr, actmn, actvar, actouts, 3, -1, {'out':1.0})

            for i in range(np.shape(self.tr)[0]):
                self.tr[i, :] = tprior['out']*self.tr[i, :]

        if lpars['mask-stimuli']:
            self.tr[:, stim_mask(t2p)] = np.nan

        # Add a boxcar filter to make it easier to see, if desired
        if not lpars['skip-boxcar']:
            for i in range(np.shape(self.tr)[0]):
                self.tr[i, :] = np.convolve(
                    self.tr[i, :], np.ones(boxcar)/boxcar, mode='same')

        # Normalize by firing rate
        firingrate = np.nansum(self.tr, 1)
        firingrate = 1.0 - firingrate/nanmax(firingrate)

        for i in range(len(firingrate)):
            self.tr[i, :] *= firingrate[i]

        # And offset the data to account for the boxcar
        if not lpars['skip-boxcar']:
            # This changes the shape, maybe just NaN out instead?
            self.tr = self.tr[:, boxcar//3:]

        self._maxframe = np.shape(self.tr)[1]
        self._maxtimem = self._maxframe/t2p.framerate/60.0
        self._tmin = self._maxtimem/self._maxframe*self._fmin

    def _getbehavior(self, t2p, lpars):
        """
        Get the behavior traces to be put along the top.
        :return: None
        """

        # Add the pupil trace and pupil mask
        pupil = t2p.pupil()
        if len(pupil) > 0:
            if lpars['mask-stimuli']:
                pupil[stim_mask(t2p)] = np.nan
            self._behtraces['pupil'] = \
                (pupil - nanmin(pupil))/(nanmax(pupil) - nanmin(pupil))
            # pmask = t2p.pupilmask(False)
            pmask = t2p.inactivity()
            if len(pmask) > 0:
                pmasked = np.copy(self._behtraces['pupil'])
                pmasked[np.invert(pmask)] = np.nan
                if lpars['mask-stimuli']:
                    pmasked[stim_mask(t2p)] = np.nan
                self._behtraces['pupil-mask'] = pmasked

        # Add brain motion
        motion = t2p.motion()
        if len(motion) > 0:
            mot = np.zeros(len(motion))
            mot[1:] = motion[1:] - motion[:-1]
            if lpars['mask-stimuli']:
                mot[stim_mask(t2p)] = np.nan
            self._behtraces['motion'] = mot

        # Add running
        running = t2p.speed()
        if len(running) > 0:
            running[running > 10] = 10
            running /= 10.0
            if lpars['mask-stimuli']:
                running[stim_mask(t2p)] = np.nan
            self._behtraces['running'] = running

        # Add licking
        licks = t2p.licking()
        if len(licks) > 0:
            if lpars['mask-stimuli']:
                mask = stim_mask(t2p)
                frames = np.arange(len(mask))
                stim_frames = frames[mask]
                licks = sorted(set(licks).difference(stim_frames))
            self._behtraces['licking'] = licks

        # Add stimuli
        corstims = {}
        errstims = {}
        for cs in ['plus', 'neutral', 'minus', 'pavlovian']:
            stims = t2p.csonsets(cs, errortrials=0)
            if len(stims) > 0:
                corstims[cs] = stims
            stims = t2p.csonsets(cs, errortrials=1)
            if len(stims) > 0:
                errstims[cs] = stims
        if corstims != {}:
            self._behtraces['correct-stimuli'] = corstims
        if errstims != {}:
            self._behtraces['error-stimuli'] = errstims

    def _gettoptrace(self, lpars, t2p, classifier, plusprob):
        """
        Prepare the contents of the top trace
        :return: None
        """

        # Only add value if top trace is ripple or photometry
        if isinstance(lpars['top-trace'], str):
            if lpars['top-trace'][:4] == 'phot':
                phot = t2p.photometry(tracetype=lpars['top-tracetype'])
                if len(phot) > 0:
                    phot -= np.min(phot)
                    phot /= np.max(phot)
                    self._toptraces['photometry'] = phot
            elif lpars['top-trace'][:3] == 'rip':
                ripple = t2p.ripple()
                if len(ripple) > 0:
                    ripple -= np.min(ripple)
                    ripple /= 80
                    ripple = np.clip(ripple, 0, 1)
                    self._toptraces['ripple'] = ripple
            elif lpars['top-trace'][:3] == 'tem':
                self._toptraces['pop-activity'] = \
                    classifier['priors']['plus']/plusprob
                print(nanmax(classifier['priors']['plus'][10000:]/plusprob),
                      plusprob, nanmax(classifier['priors']['plus'][10000:]))
            # 	popact = timeclassify.flat_population_activity(t2p.trace('deconvolved'))
            # 	self._toptraces['pop-activity'] = popact

    def _getclassifier(self, lpars, classifier, searchms, t2p):
        """
        Append the output of the classifier and account for its offset relative to the rest of the data.
        :return: None
        """
        framerate = t2p.framerate
        if lpars['display-classifier'] == 'identity':
            foffset = int(round((searchms/1000.0*framerate + 0.5)/2.0))

            for key in classifier['results']:
                if lpars['display-other'] or 'other' not in key:
                    if len(classifier['results'][key]) > 0:
                        trace = np.concatenate(
                            [[np.nan]*foffset,
                             classifier['results'][key][:-foffset]])
                        if lpars['mask-stimuli']:
                            trace[stim_mask(t2p)] = np.nan
                        self._clstraces[key] = trace
        elif lpars['display-classifier'] == 'time':
            for key in classifier['time-results']:
                if lpars['display-other'] or 'other' not in key:
                    if lpars['display-ancillary'] or 'real' in key:
                        self._clstraces[key] = np.copy(
                            classifier['time-results'][key])


def graph(path, frame, ttype, gri):
    """
    Graph
    :param path:
    :param frame:
    :param ttype:
    :return:
    """

    ts = gri.pull_time(frame)
    trs = gri.pull_traces(frame)
    cls = gri.pull_classifier(frame)
    beh = gri.pull_behavior(frame)
    bord = gri.pull_borders()
    top = gri.pull_toptrace(frame)
    clrs = pool.config.colors()

    chm = ClassifierHeatmap(
        path, ts, trs, cls, beh, bord, top, clrs, tracetype=ttype)


def grevents(classifier, t2p, md, gri, lpars, basepath, word=None):
    """
    Graph all events above a threshold
    :param classifier:
    :param gri:
    :param basepath:
    :return:
    """

    if lpars['frame'] > 0:
        path = opath.join(basepath, 'replay %s-%s-%02i unknown-%i.pdf'%(md[0], md[1], md[2], lpars['frame']))
        graph(path, lpars['frame'], lpars['display-type'])
    elif lpars['events']:
        fmin = 0 if lpars['display-training'] else t2p.lastonset()

        results = 'time-results' if lpars['display-classifier'] == 'time' else 'results'
        for cs in classifier[results]:
            print(cs)
            if 'other' not in cs and 'run' not in cs and 'rand' not in cs:
                evs = classify2p.peaks(classifier[results][cs], t2p, lpars['threshold'])
                for frame in evs:
                    if frame > fmin:
                        path = opath.join(basepath, 'replay %s-%s-%02i %s-%i.pdf'%(md[0], md[1], md[2], cs, frame))
                        graph(path, frame, lpars['display-type'])
    else:
        if word is not None:
            path = opath.join(
                basepath, '{}-{}-{}-{}-heat-classifier.png'.format(
                    md[0], md[1], md[2], word))
        else:
            path = opath.join(
                basepath,
                '%s-%s-%02i-heat-classifier.png'%(md[0], md[1], md[2]))
        graph(path, -1, lpars['display-type'], gri)

    print(path)
    return path

def deleteclassifier(pars, randomize=''):
    """
    Delete classifier after running
    :param pars: parameters from settings
    :return: None
    """

    import os

    path = paths.output(pars)
    fs = os.listdir(path)[::-1]
    out = ''

    # Change what you open whether real or random
    if len(randomize) == 0:
        for f in fs:
            if len(out) == 0:
                if f[:4] == 'real':
                    out = opath.join(path, f)
                    os.remove(out)
    else:
        for f in fs:
            if len(out) == 0:
                if f[:4] == 'rand' and f[5:5 + len(randomize)] == randomize:
                    out = opath.join(path, f)
                    os.remove(out)


def activity(andb, sated=True):
    """
    Get activity levels for temporal classifier.
    :param pars: parameters from the settings dict
    :return: baseline activity, variance of activity
    """

    # Set up temporal comparison. This is more complicated than it needs to be so that we don't recalculate the mean
    # activity every time
    actbl, actvar, actouts = -1, -1, -1

    if sated:
        atype = '-sated'
    else:
        atype = '-hungry'

    actbl = andb['activity%s-median'%atype]
    actouts = andb['activity%s-outliers'%atype]

    if actbl is None:
        actbl, actvar = 0.01, 0.08*config.default()['temporal-prior-baseline-sigma']
    else:
        actbl = actbl*config.default()['temporal-prior-baseline-sigma']
        actvar = andb['activity%s-deviation'%atype]*config.default()['temporal-prior-baseline-sigma']

    return actbl, actvar, actouts


def stim_mask(t2p, pre_pad_s=0.2, post_pad_s=0.3, post_pavlovian_pad_s=None):
    """Return a binary mask over frames of stim times."""
    mask = t2p.trialmask(
        cs='', errortrials=-1, fulltrial=False, padpre=pre_pad_s,
        padpost=post_pad_s)

    if post_pavlovian_pad_s is not None:
        pav_stim_mask = t2p.trialmask(
            cs='pavlovian', errortrials=-1, fulltrial=False,
            padpre=pre_pad_s, padpost=post_pavlovian_pad_s)
        mask = mask | pav_stim_mask

    blank_stim_mask = t2p.trialmask(
        cs='blank', errortrials=-1, fulltrial=False,
        padpre=pre_pad_s, padpost=post_pad_s)
    mask = mask & (~blank_stim_mask)

    # Not exactly sure what this is correcting for, but this will line things
    # up better
    mask = np.concatenate([[True, True], mask[:-2]])

    return mask


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Script to plot replays.""", arguments=('mice', 'dates', 'tags'))
    arg_parser.add_argument(
        "-T", "--trace_type", choices=('dff', 'deconvolved', 'raw'), default="deconvolved",
        help="Trace type to plot.")
    arg_parser.add_argument(
        "-R", "--t_range_m", nargs=2, type=int, default=(-1, 1),
        help="Time range to plot.")
    arg_parser.add_argument(
        "-D", "--display_training", action="store_true",
        help="Also plot stimulus responses, otherwise just plots spontaneous time.")
    arg_parser.add_argument(
        "-M", "--mask_stimuli", action="store_true",
        help="Mask out stimulus times from traces and classifier probability.")
    arg_parser.add_argument(
        "--new_path", action='store_true',
        help='Use new save path format: SCRIPT/mouse/cword')

    args = arg_parser.parse_args()

    return args


def main():
    lpars = {
        'sort': '',  # 'dff-no-lick', 'information'

        'display-classifier': 'identity',  # can also be 'time' or 'joint'
        'display-ancillary': False,
        'display-other': False,
        'temporal-filter': False,

        'events': False,
        'display-training': False,
        'skip-boxcar': False,
        'threshold': 0.25,
        'trange-m': (-1, 1),
        'frame': -1,

        'top-trace': 'photometry',  # can be 'ripple', 'temporal', or 'none', False
        'top-tracetype': 'dff',  # can be 'deconvolved', only applied if top-trace is photometry

        'display-type': 'deconvolved',  # can be 'dff'

        'delete-classifier': False,
        'open': False,

        'realtime': False,
    }

    args = parse_args()

    lpars['display-type'] = args.trace_type
    lpars['trange-m'] = args.t_range_m
    lpars['display-training'] = args.display_training
    lpars['mask-stimuli'] = args.mask_stimuli

    defaults = flow.config.default()

    andb = pool.database.db()

    run_types = ['training', 'spontaneous'] if args.display_training else ['spontaneous']

    dates = flow.DateSorter.frommeta(
        mice=args.mice, dates=args.dates, tags=args.tags)
    for date in dates:
        defaults['mouse'] = date.mouse
        defaults['comparison-date'] = str(date.date)
        defaults['training-date'] = str(date.date)
        for run in date.runs(run_types=run_types):
            md = (run.mouse, str(run.date), run.run, '')

            params = copy(defaults)
            params['comparison-run'] = run.run
            params['training-runs'] = flow.metadata.runs(
                run.mouse, run.date, run_types=['training'])
            params['training-other-running-runs'] = flow.metadata.runs(
                run.mouse, run.date, run_types=['running'])

            t2p = run.trace2p()
            try:
                c2p = run.classify2p()
            except classify2p.NoClassifierError as err:
                print('Unable to classify: {}_{}_{}'.format(
                    run.mouse, run.date, run.run))
                print(err)
                continue

            gri = ReplayGraphInputs(andb, params, lpars, c2p.d, t2p, md[0], md[1], params['classification-ms'],
                                    params['probability']['plus'])
            if args.new_path:
                basepath = paths.graphgroup2(
                    date.mouse, 'replay_heatmap', date.date)
                good = True
                for cs in pool.config.stimuli():
                    good = good and pool.calc.good.reactivation(date, cs)
                if not good:
                    basepath = opath.join(basepath, 'bad')
                    flow.misc.mkdir_p(basepath)
                path = grevents(
                    c2p, t2p, md, gri, lpars, basepath,
                    flow.misc.wordhash.word(params, use_new=True))
            else:
                basepath = paths.graphgroup(params, 'heatmap')
                path = grevents(c2p, t2p, md, gri, lpars, basepath)

            if lpars['delete-classifier']: deleteclassifier(params)
            if lpars['open']: getoutput('open %s' % path.replace(' ', '\\ '))


if __name__ == '__main__':
    main()
