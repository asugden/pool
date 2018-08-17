# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np

from flow import events

class ReplayFP(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            self.out['replay-fp-events-0.1-%s'%cs] = np.nan
            self.out['replay-fp-events-0.2-%s'%cs] = np.nan
            self.ev(data, cs, 0.1, '-0.1')
            self.ev(data, cs, 0.2, '-0.2')

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['replay-fp-events-0.1-%s' % (cs), 'replay-fp-events-0.2-%s' % (cs)] for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180118'

    # def trace2p(self, run):
    # 	"""
    # 	Return trace2p file, automatically injected
    # 	:param run: run number, int
    # 	:return: trace2p instance
    # 	"""

    # def classifier(self, run, randomize='', n=1):
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

    def rfp(self, data, group='sated'):
        """
        Get the value of replay minus the value of false-positive replay.
        :param data:
        :return:
        """

        cses = ['plus', 'neutral', 'minus']
        rl, rnd, nframes = {key: [] for key in cses}, {key: [] for key in cses}, {key: [] for key in cses}
        for r in data[group]:
            if np.sum(self.trace2p(r).inactivity()) > 201:
                real = self.classifier(r)
                rand = self.classifier(r, 'circshift', 10)

                gm = self.classifier(r)
                t2p = self.trace2p(r)
                fmin = t2p.lastonset()
                gmoffset = int(round(self.pars['classification-ms']/1000.0*t2p.framerate/2.0))

                for cs in cses:
                    repweight = np.copy(gm['results'][cs][fmin:-1*gmoffset])
                    repweight[repweight < 0.1] = 0

                    mask = np.invert(t2p.inactivity())[fmin:-1*gmoffset]
                    repweight[mask] = np.nan

                    if np.nansum(repweight) > 0:
                        nframes[cs].append(np.sum(np.invert(np.isnan(repweight))))
                        rl[cs].append(np.nanmean(repweight))

                        runrand = 0
                        for randcl in rand:
                            repweight = np.copy(randcl['results'][cs])
                            repweight[repweight < 0.1] = 0
                            runrand += np.nanmean(repweight)/10.0
                        rnd[cs].append(runrand)

        for cs in cses:
            rep = np.nan
            ranrep = np.nan
            if len(nframes[cs]) > 0:
                rep = np.average(rl[cs], weights=nframes[cs])
                ranrep = np.average(rnd[cs], weights=nframes[cs])
            if np.isfinite(rep):
                rep = max(0, rep - ranrep)
            if self.analysis('good-%s'%cs):
                self.out['replay-fp-%s' % cs] = rep

    def ev(self, data, cs, repmin=0.1, extratext='', group='sated', pupilmask=True):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        """

        # Average based on the number of frames used
        real, rand = [], []
        nframes = []

        for r in data[group]:
            if np.sum(self.trace2p(r).inactivity()) > 201:
                gm = self.classifier(r)
                rgms = self.classifier(r, 'circshift', 10)

                t2p = self.trace2p(r)
                fmin = t2p.lastonset()
                mask = t2p.inactivity()
                mask[:fmin] = False

                if cs in gm['results']:
                    evs = events.classpeaks(gm['results'][cs], repmin)

                    out = 0.0
                    for ev in evs:
                        if mask[ev]:
                            out += 1

                    nframes.append(np.sum(mask))
                    real.append(out/np.sum(mask)*t2p.framerate)

                    runrand = []
                    for rgm in rgms:
                        evs = events.classpeaks(rgm['results'][cs], repmin)

                        out = 0.0
                        for ev in evs:
                            if mask[ev]:
                                out += 1

                        runrand.append(out/np.sum(mask)*t2p.framerate)
                    rand.append(np.average(runrand))

        # Weighted average
        mnreal, mnrand = np.nan, np.nan
        if len(nframes) > 0:
            mnreal = np.average(real, weights=nframes)
            mnrand = np.average(rand, weights=nframes)

        if self.analysis('good-%s'%cs):
            self.out['replay-fp-events%s-%s' % (extratext, cs)] = max(0, mnreal - mnrand)
