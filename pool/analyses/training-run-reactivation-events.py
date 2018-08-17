# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np

from flow import events, parseargv

from .base import AnalysisBase


class ITIReactivations(AnalysisBase):
    def _run_analyses(self):
        out = {}
        framerate = None

        # Initialize
        for run in [2, 3, 4]:
            out['unmasked-frames-run%i'%run] = np.nan
            for cs in ['plus', 'neutral', 'minus']:
                out['unmasked-frames-run%i-after-%s' % (run, cs)] = np.nan
                for cmin in [0.1, 0.2, 0.5]:
                    out['event-peaks-run%i-%.2f-%s'%(run, cmin, cs)] = np.nan

                    for cs2 in ['plus', 'neutral', 'minus']:
                        out['event-peaks-run%i-%.2f-after-%s-%s'%(run, cmin, cs2, cs)] = np.nan


        # Run analysis
        for run in [2, 3, 4]:
            if len(self._data['train']) > 1 and run in self._data['train']:
                parameters = deepcopy(self.pars)
                parameters['training-runs'] = [r for r in self._data['train'] if r != run]
                parameters['comparison-run'] = run

                t2p = self.trace2p(run)
                gm = parseargv.classifier(parameters, '', True)

                if 'plus' not in gm['results']:
                    break

                peaks = np.zeros((3, len(gm['results']['plus'])))
                for i, cs in enumerate(['plus', 'neutral', 'minus']):
                    peaks[i, :] = gm['results'][cs]
                peaks = np.nanmax(peaks, axis=0)

                width = int(round(t2p.framerate*parameters['classification-ms']/1000.0)) + 1
                onsets = t2p.trials.astype(np.int32) - width
                onsets[onsets < 0] = 0
                offsets = t2p.offsets.astype(np.int32) + width
                framerate = t2p.framerate

                mask = np.zeros(len(gm['results']['plus'])) > 1
                mask[:onsets[0]] = 0
                if len(offsets) == len(onsets):
                    mask[offsets[-1]:] = 0
                for i, ons in enumerate(onsets):
                    if len(offsets) > i:
                        if offsets[i] < ons + t2p.framerate*2:
                            offsets[i] = ons + int(round(t2p.framerate*2)) + width
                        mask[ons:offsets[i]] = 0
                    else:
                        mask[ons:] = 0

                mask = np.bitwise_and(mask, t2p.inactivity(nostim=False))

                out['unmasked-frames-run%i'%run] = np.sum(mask == 0)

                for cs in ['plus', 'neutral', 'minus']:
                    out['unmasked-frames-run%i-after-%s' % (run, cs)] = np.sum(t2p.trialmask(cs))
                    if cs == 'plus':
                        out['unmasked-frames-run%i-after-%s'%(run, cs)] += np.sum(t2p.trialmask('pavlovian'))

                    if self.analysis('good-%s'%cs):
                        gm['results'][cs][mask] = 0
                        gm['results'][cs][gm['results'][cs] < peaks] = 0

                        for cmin in [0.1, 0.2, 0.5]:
                            out['event-peaks-run%i-%.2f-%s' % (run, cmin, cs)] = \
                                events.peaks(gm['results'][cs], t2p, cmin, max=2, downfor=2, maxlen=-1)

                            for cs2 in ['plus', 'neutral', 'minus']:
                                res = np.copy(gm['results'][cs])
                                tmask = t2p.trialmask(cs2)
                                if cs2 == 'plus':
                                    tmask = np.bitwise_or(tmask, t2p.trialmask('pavlovian'))
                                res[np.invert(tmask)] = 0
                                out['event-peaks-run%i-%.2f-after-%s-%s'%(run, cmin, cs2, cs)] = \
                                    events.peaks(res, t2p, cmin, max=2, downfor=2, maxlen=-1)

        for cs in ['plus', 'neutral', 'minus']:
            for cmin in [0.1, 0.2, 0.5]:
                numer, denom, found = 0.0, 0.0, False

                for run in self._data['train']:
                    if not np.all(np.isnan(out['event-peaks-run%i-%.2f-%s' % (run, cmin, cs)])):
                        denom += out['unmasked-frames-run%i'%run]
                        numer += len(out['event-peaks-run%i-%.2f-%s' % (run, cmin, cs)])
                        found = True

                if found:
                    out['replay-freq-iti-%.1f-%s' % (cmin, cs)] = numer/denom*framerate
                else:
                    out['replay-freq-iti-%.1f-%s'%(cmin, cs)] = np.nan

                for cs2 in ['plus', 'neutral', 'minus']:
                    numer, denom, found = 0.0, 0.0, False

                    for run in self._data['train']:
                        if not np.all(np.isnan(out['event-peaks-run%i-%.2f-after-%s-%s'%(run, cmin, cs2, cs)])):
                            denom += out['unmasked-frames-run%i-after-%s' % (run, cs2)]
                            numer += len(out['event-peaks-run%i-%.2f-after-%s-%s'%(run, cmin, cs2, cs)])
                            found = True

                    if found:
                        out['replay-freq-iti-%.1f-after-%s-%s' % (cmin, cs2, cs)] = numer/denom*framerate
                    else:
                        out['replay-freq-iti-%.1f-after-%s-%s' % (cmin, cs2, cs)] = np.nan

        return out

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    requires = ['']
    sets = ['event-peaks-run%i-%.2f-%s'%(run, cmin, cs)
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]
            for run in [2, 3, 4]] + \
           ['unmasked-frames-run%i' % run for run in [2, 3, 4]] + \
           ['event-peaks-run%i-%.2f-after-%s-%s'%(run, cmin, cs2, cs)
            for cs in ['plus', 'neutral', 'minus']
            for cs2 in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]
            for run in [2, 3, 4]] + \
           ['unmasked-frames-run%i-after-%s' % (run, cs)
            for run in [2, 3, 4]
            for cs in ['plus', 'neutral', 'minus']] + \
           ['replay-freq-iti-%.1f-%s' % (cmin, cs)
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]] + \
           ['replay-freq-iti-%.1f-after-%s-%s' % (cmin, cs2, cs)
            for cs2 in ['plus', 'neutral', 'minus']
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]]
    across = 'day'
    updated = '180725'
