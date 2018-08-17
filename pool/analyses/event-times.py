# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import events

class EventTimes(object):
    def __init__(self, data):
        self.out = {}
        for run in [5, 6, 8, 9, 10, 11, 12]:
            self.out['mask-length-run%i' % run] = np.nan
            self.out['framerate-run%i' % run] = np.nan

            for cs in ['plus', 'neutral', 'minus']:
                for cmin in np.arange(0.05, 1.01, 0.05):
                    self.out['event-peaks-run%i-%.2f-%s' % (run, cmin, cs)] = []

        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = ['event-peaks-run%i-%.2f-%s'%(run, cmin, cs)
            for cs in ['plus', 'neutral', 'minus']
            for cmin in np.arange(0.05, 1.01, 0.05)
            for run in [5, 6, 8, 9, 10, 11, 12]] + \
           [['mask-length-run%i' % run, 'framerate-run%i' % run]
            for run in [5, 6, 8, 9, 10, 11, 12]]
    across = 'day'
    updated = '180130'

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def analyze(self, data):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        :param pars:
        :param gm:
        :param t2p:
        :param cs:
        :return:
        """

        cses = ['plus', 'neutral', 'minus']
        for run in [5, 6, 8, 9, 10, 11, 12]:

            if run in data['hungry'] or run in data['sated']:
                cf = self.classifier(run)
                t2p = self.trace2p(run)
                fmin = t2p.lastonset()
                mask = t2p.inactivity()

                self.out['mask-length-run%i' % run] = np.sum(mask)
                self.out['framerate-run%i' % run] = t2p.framerate

                # Get argmaxes
                res = np.zeros((len(cses), len(cf['results'][cses[0]])))
                for j, cs2 in enumerate(cses):
                    res[j, :] = cf['results'][cs2]

                mxes = np.argmax(res, axis=0)
                for j in range(np.shape(res)[0]):
                    res[j, mxes != j] = 0

                for i, cs in enumerate(cses):
                    for cmin in np.arange(0.05, 1.01, 0.05):
                        if self.analysis('good-%s'%cs):
                            evs = events.peaks(res[i], t2p, cmin, max=2, downfor=2,
                                               maxlen=-1, fmin=fmin)

                            for ev in evs:
                                if mask[ev]:
                                    self.out['event-peaks-run%i-%.2f-%s'%(run, cmin, cs)].append(ev)
