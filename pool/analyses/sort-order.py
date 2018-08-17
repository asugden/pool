# Updated: 170414
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import math
import numpy as np

class SortOrder(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = []
    sets = ['sort-order', 'sort-borders', 'sort-simple', 'sort-simple-borders']
    across = 'day'
    updated = '180131'

    # def trace2p(self, run):
    # 	"""
    # 	Return trace2p file, automatically injected
    # 	:param run: run number, int
    # 	:return: trace2p instance
    # 	"""

    # def classifier(self, run, randomize=''):
    # 	"""
    # 	Return classifier (forced to be created if it doesn't exist), automatically injected
    # 	:param run: run number, int
    # 	:param randomize: randomization type, optional
    # 	:return:
    # 	"""

    # pars = {}  # dict of parameters, automatically injected

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def group(self, data):
        possiblecses = ['plus', 'neutral', 'minus', 'disengaged1', 'disengaged2']
        cses = [cs for cs in possiblecses if cs in data]

        order = []
        for cell in range(len(data[cses[0]])):
            mxcs = -1
            absmx = 0
            mx = 0
            for i, cs in enumerate(cses[::-1]):
                if data[cs][cell] == np.nan: data[cs][cell] = 0
                if np.isfinite(data[cs][cell]) and (mxcs < 0 or np.abs(data[cs][cell]) > absmx):
                    mxcs = i
                    mx = data[cs][cell]
                    absmx = np.abs(mx)

            order.append((mxcs, mx, cell))
        order.sort(reverse=True)

        groups = [g[0] for g in order]
        cells = [c[2] for c in order]
        borders = {}
        for i in range(len(cses)):
            if i in groups:
                borders[cses[::-1][i]] = groups.index(i)
            else:
                borders[cses[::-1][i]] = -1

        return (cells[::-1], borders)

    def simple(self, data):
        possiblecses = ['plus', 'neutral', 'minus']
        cses = [cs for cs in possiblecses if cs in data]

        order = []
        for cell in range(len(data[cses[0]])):
            mxcs = -1
            absmx = 0
            mx = 0
            for i, cs in enumerate(cses[::-1]):
                if not np.isfinite(data[cs][cell]):
                    data[cs][cell] = 0

                if data[cs][cell] > 0 and np.abs(data[cs][cell]) > absmx:
                    mxcs = i
                    mx = data[cs][cell]
                    absmx = np.abs(mx)
                elif i == 0:
                    mx = data[cs][cell]

            order.append((mxcs, mx, cell))
        order.sort(reverse=True)

        groups = [g[0] for g in order]
        cells = [c[2] for c in order]
        borders = {}
        for i in range(len(cses)):
            if i in groups:
                borders[cses[::-1][i]] = groups.index(i)
            else:
                borders[cses[::-1][i]] = -1

        borders['inhibited'] = groups.index(-1)

        return (cells[::-1], borders)

    def cslatency(self, stimdrive, peaktime, dff):
        """
        Set the sort order for latency for a specific cs
        :param cs: stimulus
        :return:
        """

        if stimdrive is None or peaktime is None or dff is None: return None

        ncells = len(dff)
        order = []
        for cell in range(ncells):
            if stimdrive[cell] > 0:
                order.append((0, peaktime[cell], cell))
            else:
                order.append((1, dff[cell], cell))
        order.sort()
        return np.array([cell[2] for cell in order])

    def latency(self, dff, sig, peak):
        """
        Get the latency sort across groups
        :param dff:
        :return:
        """

        possiblecses = ['plus', 'neutral', 'minus']
        cses = [cs for cs in possiblecses if cs in peak and peak[cs] is not None]

        order = []
        for cell in range(len(dff[cses[0]])):
            mxcs, sigcs = -1, -1
            mx, absmx = 0, 0
            siglat, sigabsmx = 0, 0

            for i, cs in enumerate(cses):
                if dff[cs][cell] == np.nan: dff[cs][cell] = 0
                if sig[cs][cell] == np.nan: sig[cs][cell] = 0
                if peak[cs][cell] == np.nan: sig[cs][cell] = 0

                if sig[cs][cell] > 0:
                    if sigcs < 0 or np.abs(dff[cs][cell]) > sigabsmx:
                        sigcs = i
                        siglat = peak[cs][cell]
                        sigabsmx = dff[cs][cell]
                else:
                    if mxcs < 0 or np.abs(dff[cs][cell]) > absmx:
                        mxcs = i
                        mx = dff[cs][cell]
                        absmx = np.abs(mx)

            if sigcs > -1:
                order.append((sigcs, siglat, cell))
            else:
                order.append((mxcs + len(cses), -mx, cell))
        order.sort()

        groups = [g[0] for g in order]
        cells = [c[2] for c in order][::-1]
        borders = {}
        for i in range(len(cses)):
            if i in groups:
                borders[cses[i]] = groups.index(i)

            if len(cses) + i in groups:
                borders['%s-2'%cses[i]] = groups.index(len(cses)+i)

        return (cells, borders)

    def copyifnotnone(self, val):
        """
        Copy if not None, otherwise return None
        :param val: any possible numpy value or None
        :return: copy of val or None
        """

        if val is None: return None
        else: return np.copy(val)

    def analyze(self, data):
        dffs = {}
        sig = {}
        peak = {}

        for cs in ['plus', 'neutral', 'minus', 'disengaged1', 'disengaged2']:
            dffs[cs] = self.copyifnotnone(self.analysis('stimulus-dff-0-2-%s'%cs))
            if np.sum(np.invert(np.isfinite(dffs[cs]))) > 4:
                dffs[cs] = self.copyifnotnone(self.analysis('stimulus-dff-all-0-2-%s'%cs))

            # sig[cs] = self.copyifnotnone(self.analysis('stimulus-drive-%s'%cs))
            # peak[cs] = self.copyifnotnone(self.analysis('peak-time-%s'%cs))

            # self.out['sort-latency-%s'%cs] = self.cslatency(sig[cs], peak[cs], dffs[cs])
        self.out['sort-order'], self.out['sort-borders'] = self.group(dffs)
        self.out['sort-simple'], self.out['sort-simple-borders'] = self.simple(dffs)
        # self.out['sort-latency'], self.out['sort-latency-borders'] = self.latency(dffs, sig, peak)
