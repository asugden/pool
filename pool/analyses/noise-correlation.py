# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class NoiseCorrelation(object):
    def __init__(self, data):
        self.out = {}
        ocorr = self.analysis('overly-correlated')

        ncs = self.get_stimuli(data)
        early = self.get_stimuli(data, endt=1)
        mid = self.get_stimuli(data, endt=1.5)
        middec = self.get_stimuli(data, endt=1.5, decon=True)
        nolick = self.get_stimuli(data, True)
        # cor = self.get_stimuli(data, False, 0)
        # err = self.get_stimuli(data, False, 1)
        for cs in ncs:
            if not isinstance(ncs[cs], float):
                ncs[cs][ocorr] = np.nan
            if not isinstance(early[cs], float):
                early[cs][ocorr] = np.nan
            if not isinstance(mid[cs], float):
                mid[cs][ocorr] = np.nan
            if not isinstance(middec[cs], float):
                middec[cs][ocorr] = np.nan
            if not isinstance(nolick[cs], float):
                nolick[cs][ocorr] = np.nan
            # if not isinstance(cor[cs], float):
            #     cor[cs][ocorr] = np.nan
            # if not isinstance(err[cs], float):
            #     err[cs][ocorr] = np.nan

            self.out['noise-correlation-%s' % cs] = ncs[cs]
            self.out['noise-correlation-0-1-%s' % cs] = nolick[cs]
            self.out['noise-correlation-0-1.5-%s' % cs] = nolick[cs]
            self.out['noise-correlation-0-1.5-decon-%s' % cs] = nolick[cs]
            self.out['noise-correlation-nolick-%s' % cs] = nolick[cs]
            # self.out['noise-correlation-correct-%s' % cs] = cor[cs]
            # self.out['noise-correlation-error-%s' % cs] = err[cs]


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['noise-correlation-%s' % cs,
             'noise-correlation-0-1-%s' % cs,
             'noise-correlation-0-1.5-%s' % cs,
             'noise-correlation-0-1.5-decon-%s' % cs,
             'noise-correlation-nolick-%s' % cs]
        #      'noise-correlation-correct-%s' % (
        # cs), 'noise-correlation-error-%s' % cs]
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180528'

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

    def get_stimuli(self, data, nolick=False, errtrial=-1, endt=2, decon=False):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        argsf = {
            'start-s': 0,
            'end-s': endt,
            'trace-type': 'dff' if not decon else 'deconvolved',
            'cutoff-before-lick-ms': -1 if not nolick else 100,
            'error-trials': errtrial,
            'baseline': (-1, 0) if not decon else (-1, -1),
        }

        val = {}
        # Go through the added stimuli and add all onsets
        for cs in ['plus', 'neutral', 'minus']:
            val[cs] = np.nan

            trs = []
            for r in data['train']:
                argsb = deepcopy(argsf)

                if len(trs) == 0:
                    trs = np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces(cs, argsb), axis=1)], axis=1)

                if cs == 'plus' and errtrial < 0:
                    trs = np.concatenate([trs, np.nanmean(self.trace2p(r).cstraces('pavlovian', argsb), axis=1)], axis=1)

            if np.shape(trs)[1] < 10:
                val[cs] = np.nan
            else:
                if not nolick:
                    val[cs] = np.corrcoef(trs)
                else:
                    dftrs = pd.DataFrame(trs.T)
                    val[cs] = dftrs.corr().as_matrix()

                nrand = 500
                ncells = np.shape(trs)[0]
                stimorder = np.arange(np.shape(trs)[1])
                for i in range(nrand):
                    for c in range(ncells):
                        np.random.shuffle(stimorder)
                        trs[c, :] = trs[c, stimorder]

                    if not nolick:
                        val[cs] -= np.corrcoef(trs)/float(nrand)
                    else:
                        dftrs = pd.DataFrame(trs.T)
                        val[cs] -= dftrs.corr().as_matrix()/float(nrand)

        return val
