# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns


class PairITIReplays(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    # NOTE: Used to pay attention to hungry and sated, now paying attention only to sated

    requires = ['classifier']
    sets = ['pair-replay-iti%s-%s'%(rtype, cs) for cs in ['plus', 'neutral', 'minus']
            for rtype in ['-0.1']] + \
           ['replay-count-iti-0.1-%s' % cs for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180718'

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

    def repev(self, data, cs, thresh=0.1, decthresh=0.2, trange=(-2, 3)):
        """
        Find all replay events
        :param cs:
        :return:
        """

        pairrep = None
        counts = None
        unpair = None
        for run in data['train']:
            evs = outfns.emptynone(self.analysis('event-peaks-run%i-%.2f-%s' % (run, thresh, cs)))

            t2p = self.trace2p(run)
            trs = t2p.trace('deconvolved')

            ncells = np.shape(trs)[0]
            if pairrep is None:
                pairrep = np.zeros((ncells, ncells))
                unpair = np.zeros((ncells, ncells))
                counts = np.zeros(ncells)

            for ev in evs:
                if -1*trange[0] < ev < np.shape(trs)[1] - trange[1]:
                    act = np.nanmax(trs[:, ev+trange[0]:ev+trange[1]], axis=1)
                    act = act > decthresh

                    actout = np.zeros((len(act), len(act)), dtype=np.bool)  # pairs of cells that overlapped
                    for i in range(len(act)):  # iterate over cells
                        actout[i, :] = np.bitwise_and(act[i], act)

                    pairrep += actout.astype(np.float64)
                    counts += act.astype(np.float64)
                    unpair += np.invert(actout).astype(np.float64)

        return pairrep, counts, unpair

    def analyze(self, data):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        :param data: data passed to analysis function
        :return: None
        """

        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            if self.analysis('good-%s'%cs):
                (self.out['pair-replay-iti-0.1-%s' % cs],
                 self.out['replay-count-iti-0.1-%s' % cs],
                 _) = self.repev(data, cs, 0.1)
            else:
                self.out['pair-replay-iti-0.1-%s' % cs] = np.nan
                self.out['replay-count-iti-0.1-%s' % cs] = np.nan