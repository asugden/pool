import math
import numpy as np

from . import base
from .. import config


class Sort(base.AnalysisBase):
    requires = []
    sets = ['sort_order', 'sort_borders']
    across = 'day'
    updated = '180831'

    def run(self, date):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        date : Date object

        Returns
        -------
        dict
            All of the output values

        """

        out = self.nanoutput()

        dffs = {}
        for cs in config.stimuli():
            stim = self.analysis('stim_dff_%s' % cs)
            dffs[cs] = np.copy(stim) if stim is not None else None
            if np.sum(np.invert(np.isfinite(dffs[cs]))) > 4:
                stim = self.analysis('stim_all_%s' % cs)
                dffs[cs] = np.copy(stim) if stim is not None else None

        out['sort_order'], out['sort_borders'] = self.simple(dffs, config.stimuli())
        return out

    @staticmethod
    def simple(dffs, preferred_order):
        """
        Return a simple sort based on preferred response category
        with an inhibited cell category at the end

        Parameters
        ----------
        dffs : dict
            dict by stimulus of numpy arrays of ncells
        preferred_order : list
            list of the preferred order of stimuli

        Returns
        -------
        list, dict
            Sorted list and the borders of between each category

        """

        cses = [cs for cs in preferred_order if cs in dffs]

        order = []
        for cell in range(len(dffs[cses[0]])):
            mxcs = -1
            absmx = 0
            mx = 0
            for i, cs in enumerate(cses[::-1]):
                if not np.isfinite(dffs[cs][cell]):
                    dffs[cs][cell] = 0

                if dffs[cs][cell] > 0 and np.abs(dffs[cs][cell]) > absmx:
                    mxcs = i
                    mx = dffs[cs][cell]
                    absmx = np.abs(mx)
                elif i == 0:
                    mx = dffs[cs][cell]

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

        try:
            borders['inhibited'] = groups.index(-1)
        except ValueError:
            borders['inhibited'] = -1

        return cells[::-1], borders
