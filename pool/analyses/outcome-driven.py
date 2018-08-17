# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import math
import numpy as np
from scipy.stats import ranksums

from replay.lib.dep import aode


class OutcomeDriven(object):
    """
    Set the log inverse p values of the visual-drivenness of cells.
    visually-driven-[cs] = -1*log(visual driven p value) where mean responses < 0 are set to 0
    """

    def __init__(self, data):
        self.out = {}

        for cs in ['ensure', 'quinine']:
            hit, miss, fr, ncells = self.stimactivity(data, cs)
            if hit is None or miss is None or fr is None:
                self.out['outcome-driven-%s'%cs] = np.zeros(ncells) > 1
                self.out['outcome-driven-anticipation-%s'%cs] = np.zeros(ncells) > 1
                self.out['outcome-driven-response-%s'%cs] = np.zeros(ncells) > 1
                self.out['outcome-driven-maxinvp-%s'%cs] = np.zeros(ncells) - 1
            else:
                self.out['outcome-driven-%s'%cs], self.out['outcome-driven-maxinvp-%s'%cs], \
                self.out['outcome-driven-anticipation-%s'%cs], self.out['outcome-driven-response-%s'%cs] = \
                    self.stats(hit, miss, cs, fr)

            self.out.pop('outcome-driven-maxinvp-%s'%cs)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['outcome-driven-%s'%cs, 'outcome-driven-anticipation-%s'%cs, 'outcome-driven-response-%s'%cs]
            for cs in ['ensure', 'quinine']]
    across = 'day'
    updated = '180524'

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

    def stimactivity(self, data, cs, gtrange=(-1, 2), ttype='dff'):
        """
        Return the activity during stimuli
        :param keep: cells to keep, vector
        :param pars: parameters used to generate classification, used for training runs
        :param gtrange: graphing time range in seconds
        :return:
        """

        graphdefs = {
            'start-s': gtrange[0],
            'end-s': gtrange[1],
            'trace-type': ttype,
        }

        hit = []
        miss = []
        fr = None
        ncells = None

        for run in data['train']:
            t2p = self.trace2p(run)
            rts = t2p.cstraces(cs, graphdefs)  # ncells, frames, nstimuli/onsets
            fts = t2p.inversecstraces(cs, graphdefs)

            if fr is None:
                fr = t2p.framerate
                ncells = np.shape(rts)[0]

            if len(rts) > 0:
                if hit == []:
                    hit = rts
                else:
                    hit = np.concatenate([hit, rts], axis=2)

            if len(fts) > 0:
                if miss == []:
                    miss = fts
                else:
                    miss = np.concatenate([miss, fts], axis=2)

        if hit == [] or np.shape(hit)[2] < 5 and cs == 'ensure':
            for run in data['train']:
                t2p = self.trace2p(run)
                rts = t2p.cstraces('pavlovian', graphdefs)  # ncells, frames, nstimuli/onsets

                if len(rts) > 0:
                    if hit == []:
                        hit = rts
                    else:
                        hit = np.concatenate([hit, rts], axis=2)

        if hit == [] or miss == [] or np.shape(hit)[2] < 5 or np.shape(miss) < 5:
            return None, None, fr, ncells

        return hit, miss, fr, ncells

    def stats(self, hits, miss, cs, fr, tintegrate=0.3, pval=0.05, ncses=2):
        """
        Get the responses integrating across different time windows
        :param hits:
        :param cs:
        :param tintegrate: integration time in seconds, will be converted to frames
        :return:
        """

        fintegrate = int(round(tintegrate*fr))

        ncells = np.shape(hits)[0]
        ntimes = int(math.floor(np.shape(hits)[1]/float(fintegrate)))

        pval /= ntimes - 1  # Correct for number of time points
        pval /= ncells  # Correct for the number of cells
        pval /= ncses  # Correct for number of CSes

        # We will save the maximum inverse p values
        vdriven = np.zeros(ncells, dtype=bool)
        vdriven_anticipation = np.zeros(ncells, dtype=bool)
        vdriven_response = np.zeros(ncells, dtype=bool)
        maxinvps = np.zeros(ncells, dtype=np.float64)

        runtot = 0
        for i in range(ntimes):
            thit = np.nanmean(hits[:, runtot:runtot + fintegrate, :], axis=1)
            tmiss = np.nanmean(miss[:, runtot:runtot + fintegrate, :], axis=1)

            for c in range(ncells):
                if np.nanmean(thit[c, :]) > 0 and np.nanmean(thit[c, :]) > np.nanmean(tmiss[c, :]):
                    pv = ranksums(tmiss[c, :], thit[c, :]).pvalue
                    logpv = -1*np.log(ranksums(tmiss[c, :], thit[c, :]).pvalue)
                    if logpv > maxinvps[c]: maxinvps[c] = logpv
                    if pv <= pval:
                        vdriven[c] = True

                    if i < 3 and pv <= pval:
                        vdriven_anticipation[c] = True
                    elif i >= 3 and pv <= pval:
                        vdriven_response[c] = True

            runtot += fintegrate

        return vdriven, maxinvps, vdriven_anticipation, vdriven_response

