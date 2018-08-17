# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np

from flow import outfns

from .base import AnalysisBase


class CosdistStimuli(AnalysisBase):
    def _run_analyses(self):
        out = {}

        framerate = self.analysis('framerate-run%i' % self._data['sated'][0])
        keep = np.invert(np.bitwise_or(self.analysis('activity-outliers'), self.analysis('activity-sated-outliers')))
        proto = outfns.protovectors(self._data['mouse'], self._data['date'], keep=keep, hz=framerate)

        ens, quin = self.outcome()

        for cs in ['plus', 'neutral', 'minus']:
            for stimtype in ['dff-0-1', 'dff-0-2', 'decon-0-1', 'decon-0-2']:
                mnstim = self.analysis('stimulus-%s-%s' % (stimtype, cs))[keep]

                out['cosdist-stim-ensure-%s-%s' % (stimtype, cs)] = outfns.cosinedist(proto['ensure'], mnstim)
                out['cosdist-stim-quinine-%s-%s' % (stimtype, cs)] = outfns.cosinedist(proto['quinine'], mnstim)

            stimtype = 'decon-resp-0-1'
            if len(ens) > 0:
                mnstim = self.analysis('stimulus-decon-0-1-%s'%(cs))[keep]
                out['cosdist-stim-ensure-%s-%s' % (stimtype, cs)] = outfns.cosinedist(ens[keep], mnstim)
            else:
                out['cosdist-stim-ensure-%s-%s'%(stimtype, cs)] = np.nan

            if len(quin) > 0:
                mnstim = self.analysis('stimulus-decon-0-1-%s'%(cs))[keep]
                out['cosdist-stim-quinine-%s-%s' % (stimtype, cs)] = outfns.cosinedist(quin[keep], mnstim)
            else:
                out['cosdist-stim-quinine-%s-%s'%(stimtype, cs)] = np.nan

        return out

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    requires = ['']
    sets = [['cosdist-stim-ensure-%s-%s' % (stimtype, cs), 'cosdist-stim-quinine-%s-%s' % (stimtype, cs)]
            for stimtype in ['dff-0-1', 'dff-0-2', 'decon-0-1', 'decon-0-2', 'decon-resp-0-1']
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180508'

    def outcome(self):
        args = {
            'start-s': 0,
            'end-s': 1,
            'trace-type': 'deconvolved',
        }

        es, qs = [], []

        for run in self._data['train']:
            t2p = self.trace2p(run)
            trs = t2p.cstraces('ensure', deepcopy(args))
            if es == []:
                es = trs
            else:
                es = np.concatenate([es, trs], axis=2)

            trs = t2p.cstraces('quinine', deepcopy(args))
            if qs == []:
                qs = trs
            else:
                qs = np.concatenate([qs, trs], axis=2)

        es = np.nanmean(es, axis=1)
        qs = np.nanmean(qs, axis=1)

        es = np.nanmean(es, axis=1)
        qs = np.nanmean(qs, axis=1)

        return es, qs


