# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

# from copy import deepcopy
import numpy as np

from replay.lib import classify
from flow import events
# from scipy import stats

class TemporalPrior(object):
    """
    Set the log inverse p values of the visual-drivenness of cells.
    visually-driven-[cs] = -1*log(visual driven p value) where mean responses < 0 are set to 0
    """

    def __init__(self, data):
        self.out = {}
        self.analyze(data)

        self.out['hungry-mask-length'] = 0
        for run in data['hungry']:
            t2p = self.trace2p(run)
            self.out['hungry-mask-length'] += np.nansum(t2p.inactivity())

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['temporal-prior', 'temporal-prior-events-0.1', 'temporal-prior-events-0.2', 'temporal-prior-events-0.3',
            'temporal-prior-events-0.4', 'temporal-prior-events-0.5', 'temporal-prior-events-0.6',
            'temporal-prior-events-0.7', 'temporal-prior-events-0.8', 'temporal-prior-events-0.9',
            'mask-length', 'hungry-mask-length', 'population-activity']
    across = 'day'
    updated = '180501'

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

    def checkfraction(self, run, trs, mask):
        """
        Commented out. Check the fraction of reactivation events found at different temporal threshold levels

        :param run:
        :param trs:
        :param mask:
        :return:
        """

        actbl = self.analysis('activity-sated-median')
        actvar = self.analysis('activity-sated-deviation')
        actouts = self.analysis('activity-sated-outliers')
        tpriors = classify.temporal_prior(trs, actbl, actvar, actouts, 4, -1, {'out': 1})

        for threshold in np.arange(0.1, 0.6, 0.1):
            ons, offs = events.count(tpriors['out'], threshold, offsets=True)
            nframes = np.shape(trs)[1]
            tevents = np.zeros(nframes)
            for on, off in zip(ons, offs):
                tevents[on:off] = 1

            overlaps = 0
            total = 0
            for cs in ['plus', 'neutral', 'minus']:
                evs = self.analysis('event-peaks-run%i-%.2f-%s' % (run, 0.1, cs))
                if evs is not None and len(evs) > 0:
                    total += len(evs)
                    overlaps += np.sum(tevents[np.array(evs)])

            fp = open('counts-%.1f.txt' % threshold, 'a')
            fp.write('%i\t%i\n' % (overlaps, total))
            fp.close()

    def traces(self, data):
        """
        Get the amount of total running and the running per CS
        :param data:
        :return:
        """

        out = []
        for r in data['sated']:
            t2p = self.trace2p(r)
            trs = t2p.trace('deconvolved')
            mask = t2p.inactivity()
            fmin = t2p.lastonset()
            mask[:fmin] = False

            # self.checkfraction(r, trs, mask)

            if len(out) == 0: out = trs[:, mask]
            else: out = np.concatenate([out, trs[:, mask]], axis=1)

        return out, t2p.framerate

    def analyze(self, data):
        trs, hz = self.traces(data)
        self.out['mask-length'] = np.shape(trs)[1]
        self.out['population-activity'] = np.nanmean(trs)
        actbl = self.analysis('activity-sated-median')
        actvar = self.analysis('activity-sated-deviation')
        actouts = self.analysis('activity-sated-outliers')

        if actbl is None or actvar is None:
            self.out['temporal-prior'] = None
            self.out['temporal-prior-events'] = None
        else:
            actvar *= 3.0

            tpriors = classify.temporal_prior(trs, actbl, actvar, actouts, 4, -1, {'out':1})
            self.out['temporal-prior'] = np.nanmean(tpriors['out'])

            for i in np.arange(0.1, 0.91, 0.1):
                self.out['temporal-prior-events-%.1f' % i] = float(len(events.count(tpriors['out'], i)))/\
                                                                len(tpriors['out'])*hz

