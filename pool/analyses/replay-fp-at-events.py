# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np

from flow import events

class ReplayFPEvents(object):
    def __init__(self, data):
        self.out = {}
        for run in [5, 6, 7, 8, 9, 10, 11, 12]:
            for rtype in ['circshift', 'identity']:
                for cs in ['plus', 'neutral', 'minus']:
                    for cmin in np.arange(0.05, 1.01, 0.05):
                        self.out['rand-%s-evs-%0.2f-run%i-%s'%(rtype, cmin, run, cs)] = []

                if run in data['sated']: # or run in data['hungry']:
                    self.ev(data, run, rtype)



    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = ['rand-%s-evs-%0.2f-run%i-%s' % (rtype, cmin, run, cs)
            for cs in ['plus', 'neutral','minus']
            for rtype in ['circshift', 'identity']
            for cmin in np.arange(0.05, 1.01, 0.05)
            for run in [5, 6, 7, 8, 9, 10, 11, 12]]
    across = 'day'
    updated = '180203'

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

    def ev(self, data, run, rtype='circshift'):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        """

        # Average based on the number of frames used
        real, rand = [], []
        nframes = []

        cses = ['plus', 'neutral', 'minus']
        t2p = self.trace2p(run)

        if np.sum(t2p.inactivity()) > 201:
            # gm = self.classifier(run)
            rgms = self.classifier(run, rtype, 10)

            fmin = t2p.lastonset()
            mask = t2p.inactivity()
            mask[:fmin] = False

            for rgm in rgms:
                res = np.zeros((len(cses), len(rgm['results'][cses[0]])))
                for j, cs2 in enumerate(cses):
                    res[j, :] = rgm['results'][cs2]

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
                                    self.out['rand-%s-evs-%0.2f-run%i-%s'%(rtype, cmin, run, cs)].append(ev)

