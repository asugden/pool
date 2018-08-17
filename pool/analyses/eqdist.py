# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns


class EventEQDists(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            for minc in [0.1, 0.2, 0.5]:
                for run in [5, 6, 8, 9, 10, 11, 12]:
                    self.out['eqdist-run%i-%.1f-%s' % (run, minc, cs)] = []
                    self.out['edist-run%i-%.1f-%s' % (run, minc, cs)] = []
                    self.out['qdist-run%i-%.1f-%s' % (run, minc, cs)] = []

        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['eqdist-run%i-%.1f-%s' % (run, minc, cs),
             'edist-run%i-%.1f-%s' % (run, minc, cs),
             'qdist-run%i-%.1f-%s' % (run, minc, cs),]
            for cs in ['plus', 'neutral', 'minus']
            for minc in [0.1, 0.2, 0.5]
            for run in [5, 6, 8, 9, 10, 11, 12]]
    across = 'day'
    updated = '180203'

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

        frange = (-2, 3)

        framerate = self.analysis('framerate-run%i' % data['sated'][0])
        keep = np.invert(np.bitwise_or(self.analysis('activity-outliers'), self.analysis('activity-sated-outliers')))
        proto = outfns.protovectors(data['mouse'], data['date'], keep=keep, hz=framerate)

        for run in [5, 6, 8, 9, 10, 11, 12]:
            if run in data['hungry'] or run in data['sated']:
                for cmin in [0.1, 0.2, 0.5]:
                    for cs in ['plus', 'neutral', 'minus']:
                        if self.analysis('good-%s'%cs):
                            t2p = self.trace2p(run)
                            trs = t2p.trace('deconvolved')

                            evs = self.analysis('event-peaks-run%i-%.2f-%s' % (run, cmin, cs))
                            if evs is not None and len(evs) > 0:
                                for ev in evs:
                                    act = np.nanmean(trs[:, ev + frange[0]: ev + frange[1]], axis=1)
                                    eqd = outfns.eqdist(proto, act[keep], cs)
                                    ed = outfns.edist(proto, act[keep], cs)
                                    qd = outfns.qdist(proto, act[keep], cs)

                                    self.out['eqdist-run%i-%.1f-%s' % (run, cmin, cs)].append(eqd)
                                    self.out['edist-run%i-%.1f-%s' % (run, cmin, cs)].append(ed)
                                    self.out['qdist-run%i-%.1f-%s' % (run, cmin, cs)].append(qd)

