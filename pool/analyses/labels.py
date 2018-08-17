# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
from scipy.stats import ttest_rel
import warnings

class HungrySated(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'neutral', 'minus']:
            self.cstuned(data, cs)
        self.eventtuned(data, 'licking')
        self.eventtuned(data, 'ensure')
        self.eventtuned(data, 'quinine')

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = []
    sets = [['modulated-by-%s'%cs, 'modulation-by-%s'%cs] for cs in ['plus-error', 'plus-correct', 'neutral-error',
                                                                      'neutral-correct', 'minus-error', 'minus-correct',
                                                                      'ensure', 'quinine', 'licking']]
    across = 'day'
    updated = '170810'

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

    def lickmask(self, tr, ons, flicks, emptyframes=16, prelickframes=2):
        """
        Mask an array to account for
        :param tr: traces, matrix of ncells by nframes
        :param ons: cs onsets
        :param flicks: cs first licks
        :param maskframes: how many frames to mask after first lick
        :return: masked traces, matrix ncells by nframes
        """

        ons.append(np.shape(tr)[1]+1)
        for onset, nextons, lick in zip(ons[:-1], ons[1:], flicks):
            if np.isfinite(lick) and lick > 0:
                lick = int(lick) + onset
                tr[:, lick-prelickframes:nextons-emptyframes] = np.nan
        return tr

    def cstuned(self, data, cs):
        """
        Get the hungry values and account for the median lick time.
        :param data:
        :return:
        """

        # Set up the arguments to pass to cstraces
        args = {'start-s': 0, 'end-s': 0.5, 'trace-type': 'dff', 'cutoff-before-lick-ms': 100, 'baseline': (-1, 0)}

        for err in [0, 1]:
            p, val = [], []
            for off in [0, 0.5, 1.0, 1.5]:
                g1, g2 = [], []

                # Get sated stimulus days
                for r in data['train']:
                    t2p = self.trace2p(r)
                    trs = np.copy(t2p.trace('dff'))
                    blrange = (int(round(-0.5*t2p.framerate)), 0)
                    frange = (int(round(off*t2p.framerate)), int(round((off+0.5)*t2p.framerate)))
                    ons = t2p.csonsets(cs, errortrials=err)

                    if cs == ['plus']:
                        ons = np.concatenate([ons, t2p.csonsets('pavlovian')])

                    flicks = t2p.firstlick(cs, errortrials=err)
                    trs = self.lickmask(trs, ons, flicks, int(round(t2p.framerate)), int(round(t2p.framerate*0.1)))

                    for onset in ons:
                        if np.sum(np.isfinite(trs[:, onset+frange[0]:onset+frange[1]])) > 1:
                            g2.append(np.nanmean(trs[:, onset+frange[0]:onset+frange[1]], axis=1))
                            g1.append(np.nanmean(trs[:, onset + blrange[0]:onset + blrange[1]], axis=1))

                if len(g2) == len(g1) > 2:
                    g1, g2 = np.array(g1), np.array(g2)

                    mng1 = np.nanmean(g1, axis=0)
                    mng2 = np.nanmean(g2, axis=0)

                    nans = np.logical_or(np.invert(np.isfinite(mng1)),  np.invert(np.isfinite(mng2)))

                    p1 = np.ones(np.shape(g1)[1])
                    for c in range(len(mng1)):
                        p1[c] = ttest_rel(g1[:, c], g2[:, c])[1]/2.0*4.0  # Correct for 1 tail and also 4
                        # separate tests

                    mng1[nans], mng2[nans] = 0.0, 0.0
                    p1[np.array(mng2) < np.array(mng1)] = 1.0
                    p1[np.logical_or(nans, np.invert(np.isfinite(p1)))] = 1.0
                    if len(p) == 0:
                        p = p1
                        val = mng2 - mng1
                    else:
                        val[p1 < p] = (mng2 - mng1)[p1 < p]
                        p[p1 < p] = p1[p1 < p]


            errcode = 'error' if err == 1 else 'correct'
            self.out['modulated-by-%s-%s'%(cs, errcode)] = p
            self.out['modulation-by-%s-%s'%(cs, errcode)] = val

    def eventtuned(self, data, event):
        """
        Get the p values and responses to licking
        :param self:
        :param data:
        :param event: str, 'licking', 'quinine', or 'ensure'
        :return:
        """

        g1, g2 = [], []
        for r in data['train']:
            t2p = self.trace2p(r)
            trs = np.copy(t2p.trace('dff'))
            blrange = (int(round(-0.5*t2p.framerate)), 0)
            frange = (0, int(round(0.5*t2p.framerate)))

            if event == 'licking':
                ons = t2p.licking()
            elif event == 'ensure':
                ons = t2p.ensure()
            elif event == 'quinine':
                ons = t2p.quinine()

            for onset in ons[ons > 0]:
                if -1*blrange[0] < onset < np.shape(trs)[1] - frange[1]:
                    g2.append(np.nanmean(trs[:, onset + frange[0]:onset + frange[1]], axis=1))
                    g1.append(np.nanmean(trs[:, onset + blrange[0]:onset + blrange[1]], axis=1))

        g1, g2 = np.array(g1), np.array(g2)
        mng1 = np.nanmean(g1, axis=0)
        mng2 = np.nanmean(g2, axis=0)
        nans = np.logical_or(np.invert(np.isfinite(mng1)), np.invert(np.isfinite(mng2)))

        if np.shape(g1)[0] < 4:
            self.out['modulated-by-%s'%(event)] = None
            self.out['modulation-by-%s'%(event)] = None
        else:
            p1 = np.ones(np.shape(g1)[1])
            for c in range(len(mng1)):
                p1[c] = ttest_rel(g1[:, c], g2[:, c])[1]/2.0  # Correct for 1 tail

            mng1[nans], mng2[nans] = 0.0, 0.0
            p1[np.array(mng2) < np.array(mng1)] = 1.0
            p1[np.logical_or(nans, np.invert(np.isfinite(p1)))] = 1.0

            self.out['modulated-by-%s'%(event)] = p1
            self.out['modulation-by-%s'%(event)] = mng2 - mng1