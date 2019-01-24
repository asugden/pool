from builtins import object, range

from copy import deepcopy
import math
import numpy as np

from flow import paths

class StimulusDFF(object):
    _pars = {
        'trange-ms': (-1000, 8000),
        'baseline-ms': (-1000, 0),
        'framerate': -1,
        'order-type': 'traditional',
    }

    _order = []
    _borders = {}
    _data = {}
    _orderedcses = []
    _tracetype = ''

    def __init__(self):
        pass

    def onsets(self, t2p, lickprems=100, tracetype='dff', errortrials=-1, orderedcses=['plus', 'neutral', 'minus'],
        equivalents={'plus':'pavlovian'}):
        self._pars['framerate'] = t2p.framerate
        self._timetoframes()

        for cs in orderedcses:
            if cs not in self._orderedcses:
                self._orderedcses.append(cs)

        # Set up the fluorescence and baseline arguments for passing
        argsf = {
            'start-s': self._pars['trange-ms'][0]/1000.,
            'end-s': self._pars['trange-ms'][1]/1000.,
            'trace-type': tracetype,
            'cutoff-before-lick-ms': lickprems,
            'error-trials': errortrials,
        }

        self._tracetype = tracetype

        argsb = deepcopy(argsf)
        argsb['start-s'] = self._pars['baseline-ms'][0]/1000.
        argsb['end-s'] = self._pars['baseline-ms'][1]/1000.

        # Go through the added stimuli and add all onsets
        for cs in orderedcses:
            out = t2p.cstraces(cs, argsf) # ncells, frames, nstimuli/onsets
            # for fr in range(np.shape(out)[1]):
            #   out[:, fr, :] -= np.nanmean(t2p.cstraces(cs, argsb), 1)

            if cs in equivalents:
                eq = t2p.cstraces(equivalents[cs], argsf)
                # for fr in range(np.shape(eq)[1]):
                #   eq[:, fr, :] -= np.nanmean(t2p.cstraces(equivalents[cs], argsb), 1)
                out = np.concatenate((out, eq), axis=2)

            if cs not in self._data: self._data[cs] = out
            else:
                self._data[cs] = np.concatenate((self._data[cs], out), axis=2)

    def add(self, t2p, orderedcses=['plus', 'neutral', 'minus'], equivalents={'plus':'pavlovian'}):
        self._pars['framerate'] = t2p.framerate
        self._timetoframes()
        totalframes = self._pars['trange-frames'][1] - self._pars['trange-frames'][0]

        for cs in orderedcses:
            if cs not in self._orderedcses:
                self._orderedcses.append(cs)

        dff = t2p.trace('dff')
        for cs in orderedcses:
            ons = t2p.csonsets(cs)
            if cs in equivalents:
                ons.extend(t2p.csonsets(equivalents[cs]))

            off = 0
            if cs not in self._data: self._data[cs] = np.zeros((np.shape(dff)[0], 
                totalframes, len(ons)))
            else:
                former = self._data[cs]
                off = np.shape(former)[2]
                self._data[cs] = np.zeros((np.shape(dff)[0], totalframes, len(ons) + off))
                self._data[cs][:, :, :off] = former

            for i in range(len(ons)):
                self._data[cs][:, :, off+i] = dff[:, ons[i] + self._pars['trange-frames'][0]:
                    ons[i] + self._pars['trange-frames'][1]]

    def matrix9(self, pars={}):
        """
        Get a sorted matrix of all of all combinations of cs and 
        cs-responsive cells.
        """

        for p in pars: self._pars[p] = pars[p]
        self._timetoframes()

        first = self._orderedcses[0]
        ncells = np.shape(self._data[first])[0]
        nframes = np.shape(self._data[first])[1]
        out = np.zeros((ncells, nframes*len(self._orderedcses)))

        for i, cs in enumerate(self._orderedcses):
            out[:, i*nframes:(i+1)*nframes] = np.nanmean(self._data[cs], 2)

            if self._tracetype == 'dff':
                bl1 = self._pars['baseline-frames'][0] - self._pars['trange-frames'][0]
                bl2 = self._pars['baseline-frames'][1] - self._pars['baseline-frames'][0]
                
                baseline = np.nanmean(out[:, i*nframes + bl1:i*nframes + bl1 + bl2], 1)
                for f in range(nframes):
                    out[:, i*nframes + f] -= baseline

        return out

    def trials(self, cnum, pars={}):
        """
        Get a sorted matrix of individual trials of the cell index specified.
        :param pars:
        :return:
        """

        for p in pars: self._pars[p] = pars[p]
        self._timetoframes()

        first = self._orderedcses[0]
        ncells = np.shape(self._data[first])[0]
        nframes = np.shape(self._data[first])[1]

        ntrials = np.sum([np.shape(self._data[cs])[2] for cs in self._orderedcses])
        out = np.zeros((ntrials, nframes))

        ltrial = 0
        borders = {}
        for i, cs in enumerate(self._orderedcses):
            ttrial = np.shape(self._data[cs])[2]
            out[ltrial:ltrial + ttrial, :] = self._data[cs][cnum, :, :].T
            borders[cs] = ltrial
            ltrial += ttrial

        if self._tracetype == 'dff':
            bl1 = self._pars['baseline-frames'][0] - self._pars['trange-frames'][0]
            bl2 = self._pars['baseline-frames'][1] - self._pars['baseline-frames'][0]

            baseline = np.nanmean(out[:, bl1:bl1 + bl2], 1)
            for f in range(nframes):
                out[:, f] -= baseline

        return out, borders

    def traditional(self):
        """
        Group into plus, neutral, minus by absolute magnitude of
        response, then group within by response. Returns the sorting
        and the first cell of each group
        """

        groupmeans = {}
        for cs in self._orderedcses:
            groupmeans[cs] = np.mean(self._data[cs], 1)

        order = []
        for cell in range(len(groupmeans[self._orderedcses[0]])):
            mxcs = -1
            absmx = 0
            mx = 0
            for i, cs in enumerate(self._orderedcses):
                if mxcs < 0 or np.abs(groupmeans[cs][cell]) > absmx:
                    mxcs = i
                    mx = groupmeans[cs][cell]
                    absmx = np.abs(mx)
            
            order.append((mxcs, mx, cell))
        order.sort()

        groups = [g[0] for g in order]
        cells = [c[2] for c in order]
        borders = []
        for i in range(len(self._orderedcses)):
            if i in groups:
                borders.append(groups.index(i))
            else:
                borders.append(-1)

        return (cells, borders, self._orderedcses)


    def _timetoframes(self):
        """
        Convert timing from parameters into number of frames to integrate
        """

        self._pars['trange-frames'] = [int(math.floor(fr*self._pars['framerate']/1000.0)) 
            for fr in self._pars['trange-ms']]
        self._pars['baseline-frames'] = [int(math.floor(fr*self._pars['framerate']/1000.0)) 
            for fr in self._pars['baseline-ms']]

    def reset(self):
        """
        Reset values. Shouldn't be necessary, weirdly is
        :return:
        """
        self._pars = {
            'trange-ms': (-1000, 8000),
            'baseline-ms': (-1000, 0),
            'framerate': -1,
            'order-type': 'traditional',
        }

        self._order = []
        self._borders = {}
        self._data = {}
        self._orderedcses = []
        self._tracetype = ''

def sdff():
    obt = StimulusDFF()
    return obt

def dff(pars, lpars, runs=[]):
    default = {
        'trange-ms': (-1000, 8000),
        'baseline-ms': (-1000, 0),

        'trace-type': 'deconvolved',
        'cutoff-before-lick-ms': -1,
        'error-trials': -1,
        'ordered-cses': ['plus', 'neutral', 'minus'],
        'equivalents': {'plus': 'pavlovian'},
    }

    for p in lpars: default[p] = lpars[p]

    sdff = StimulusDFF()
    sdff.reset()
    sdff._pars['trange-ms'] = default['trange-ms']
    sdff._pars['baseline-ms'] = default['baseline-ms']

    mouse = pars['mouse']
    date = pars['training-date']
    if len(runs) == 0: runs = pars['training-runs']
    for run in runs:
        t2p = paths.gett2p(mouse, date, run)
        sdff.onsets(t2p, default['cutoff-before-lick-ms'], default['trace-type'], default['error-trials'],
            default['ordered-cses'], default['equivalents'])
    return (sdff.matrix9(), t2p.framerate)

def trials(pars, lpars, cnum):
    default = {
        'trange-ms': (-1000, 8000),
        'baseline-ms': (-1000, 0),

        'trace-type': 'deconvolved',
        'cutoff-before-lick-ms': -1,
        'error-trials': -1,
        'ordered-cses': ['plus', 'neutral', 'minus'],
        'equivalents': {'plus': 'pavlovian'},
    }

    for p in lpars: default[p] = lpars[p]

    sdff = StimulusDFF()
    sdff.reset()
    sdff._pars['trange-ms'] = default['trange-ms']
    sdff._pars['baseline-ms'] = default['baseline-ms']

    mouse = pars['mouse']
    date = pars['training-date']
    for run in pars['training-runs']:
        t2p = paths.gett2p(mouse, date, run)
        sdff.onsets(t2p, default['cutoff-before-lick-ms'], default['trace-type'], default['error-trials'],
            default['ordered-cses'], default['equivalents'])
    hm, borders = sdff.trials(cnum)
    return (hm, borders, t2p.framerate)
