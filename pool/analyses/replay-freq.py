# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns

class ReplayFreq(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['replay-freq-%.1f-%s'%(cmin, cs), 'replay-freq-hungry-%.1f-%s'%(cmin, cs)]
            for cs in ['plus', 'neutral', 'minus']
            for cmin in [0.1, 0.2, 0.5]] + \
           [['replay-freq-0.05-%s'%(cs), 'replay-freq-hungry-0.05-%s'%(cs)]
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180202'

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

    def ev(self, data, cs, repmin, group='sated'):
        """
        Double-checking the output because of some unrelated odd results
        :param data:
        :param cs:
        :param repmin:
        :param group:
        :return:
        """

        reps = 0.0
        nframes = 0.0
        framerate = 0.0

        for run in data[group]:
            evs = outfns.emptynone(self.analysis('event-peaks-run%i-%.2f-%s' % (run, repmin, cs)))
            reps += len(evs)
            nframes += outfns.zeronone(self.analysis('mask-length-run%i' % run))

            if framerate == 0 and self.analysis('framerate-run%i' % run) is not None:
                framerate = self.analysis('framerate-run%i' % run)

        if nframes == 0:
            return np.nan
        else:
            return reps/nframes*framerate

    def analyze(self, data):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        :param pars:
        :param gm:
        :param t2p:
        :param cs:
        :return:
        """

        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            good = self.analysis('good-%s' % cs)
            for cmin in [0.1, 0.2, 0.5]:
                if good:
                    self.out['replay-freq-%.1f-%s' % (cmin, cs)] = self.ev(data, cs, cmin)
                    self.out['replay-freq-hungry-%.1f-%s' % (cmin, cs)] = self.ev(data, cs, cmin, 'hungry')

                else:
                    self.out['replay-freq-%.1f-%s'%(cmin, cs)] = np.nan
                    self.out['replay-freq-hungry-%.1f-%s'%(cmin, cs)] = np.nan

            cmin = 0.05
            if good:
                self.out['replay-freq-%.2f-%s'%(cmin, cs)] = self.ev(data, cs, cmin)
                self.out['replay-freq-hungry-%.2f-%s'%(cmin, cs)] = self.ev(data, cs, cmin, 'hungry')

            else:
                self.out['replay-freq-%.2f-%s'%(cmin, cs)] = np.nan
                self.out['replay-freq-hungry-%.2f-%s'%(cmin, cs)] = np.nan
