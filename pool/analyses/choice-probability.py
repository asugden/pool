# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np
from scipy import stats
from sklearn import metrics
import warnings

class ChoiceProbability(object):
    def __init__(self, data):
        self.out = {}

        for cs in ['plus', 'neutral', 'minus']:
            for t in [(0, 1), (0, 2)]:
                self.out['chprob-%i-%i-%s'%(t[0], t[1], cs)], self.out['chprob-p-%i-%i-%s'%(t[0], t[1], cs)], \
                    self.out['ncorrect-%s'%cs], self.out['nfalse-%s'%cs] = self.auroc_error(data, cs, t)

            nolick, _, _, _ = self.auroc_error(data, cs, (0, 1), nolick=True)

            self.out['abschprob-%s'%(cs)] = 2.0*np.abs(np.array(self.out['chprob-0-2-%s'%(cs)]) - 0.5)
            self.out['abschprob-nolick-%s'%(cs)] = 2.0*np.abs(np.array(nolick) - 0.5)

            if cs == 'plus':
                self.out['chprob-gopos-plus'] = 1.0 - np.array(self.out['chprob-0-2-plus'])
                self.out['chprob-gopos-nolick-plus'] = 1.0 - np.array(nolick)
            else:
                self.out['chprob-gopos-%s'%cs] = np.array(self.out['chprob-0-2-%s'%cs])
                self.out['chprob-gopos-nolick-%s'%cs] = np.array(nolick)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = []
    sets = [[
                'ncorrect-%s'%cs,
                'nfalse-%s'%cs,
                'abschprob-%s'%cs,
                'abschprob-nolick-%s'%cs,
                'chprob-gopos-%s'%cs,
                'chprob-gopos-nolick-%s'%cs,
                [['chprob-%i-%i-%s'%(t[0], t[1], cs), 'chprob-p-%i-%i-%s'%(t[0], t[1], cs)] for t in [(0, 1), (0, 2)]],
            ] for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180206'

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

    def auroc_error(self, data, cs, trange_s=(0, 2), nolick=False):
        """
        Find the cell-by-cell response to hits vs misses
        :param pars:
        :return: vector of hits - misses/mean
        """

        csargs = {
            'start-s': trange_s[0],
            'end-s': trange_s[1],
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1 if not nolick else 100,
        }

        hits = None
        miss = None

        for r in data['train']:
            t2p = self.trace2p(r)

            # Of shape ncells, frames, nstimuli/onsets.
            resp = t2p.cstraces(cs, csargs)

            if np.shape(resp)[2] > 0:
                errcodes = t2p.errors(cs)
                succodes = np.invert(errcodes)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    resp = np.nanmean(resp, axis=1)
                if hits is None:
                    hits = resp[:, succodes]
                    miss = resp[:, errcodes]
                else:
                    hits = np.concatenate([hits, resp[:, succodes]], axis=1)
                    miss = np.concatenate([miss, resp[:, errcodes]], axis=1)

        # Make sure that this is possible to compute
        if hits is None or miss is None: return ([], [], np.nan, np.nan)

        nhits = np.shape(hits)[1]
        nmiss = np.shape(miss)[1]
        ncell = np.shape(hits)[0]
        all = np.concatenate([hits, miss], axis=1)
        classes = np.array([0]*nhits + [1]*nmiss)

        # Make sure that this is possible to compute
        if nhits == 0 or nmiss == 0: return ([], [], np.nan, np.nan)

        # Get the auROC for each cell
        auroc = np.zeros(ncell)
        pvals = np.zeros(ncell)
        for c in range(ncell):
            nnanclasses = classes[np.invert(np.isnan(all[c]))]
            nnanall = all[c, np.invert(np.isnan(all[c]))]
            if len(np.unique(nnanclasses)) < 2:
                auroc[c] = np.nan
                pvals[c] = np.nan
            else:
                auroc[c] = metrics.roc_auc_score(nnanclasses, nnanall)
                pvals[c] = stats.ranksums(hits[c, np.invert(np.isnan(hits[c, :]))],
                                          miss[c, np.invert(np.isnan(miss[c, :]))]).pvalue

        return (auroc, pvals, nhits, nmiss)

    def auroc_hit_miss(self, data, trange_s=(0, 2)):
        """
        Find the cell-by-cell response to hits vs misses
        :param pars:
        :return: vector of hits - misses/mean
        """

        csargs = {
            'start-s': trange_s[0],
            'end-s': trange_s[1],
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1,
        }

        hits = None
        miss = None

        for r in data['train']:
            t2p = self.trace2p(r)

            # Of shape ncells, frames, nstimuli/onsets.
            resp = t2p.cstraces('plus', csargs)

            if np.shape(resp)[2] > 0:
                errcodes = t2p.errors('plus')
                succodes = np.invert(errcodes)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    resp = np.nanmean(resp, axis=1)
                if hits is None:
                    hits = resp[:, succodes]
                    miss = resp[:, errcodes]
                else:
                    hits = np.concatenate([hits, resp[:, succodes]], axis=1)
                    miss = np.concatenate([miss, resp[:, errcodes]], axis=1)

        # Make sure that this is possible to compute
        if hits is None or miss is None: return ([], [], np.nan, np.nan)

        nhits = np.shape(hits)[1]
        nmiss = np.shape(miss)[1]
        ncell = np.shape(hits)[0]
        all = np.concatenate([hits, miss], axis=1)
        classes = np.array([0]*nhits + [1]*nmiss)

        # Make sure that this is possible to compute
        if nhits == 0 or nmiss == 0: return ([], [], np.nan, np.nan)

        # Get the auROC for each cell
        auroc = np.zeros(ncell)
        pvals = np.zeros(ncell)
        for c in range(ncell):
            nnanclasses = classes[np.invert(np.isnan(all[c]))]
            nnanall = all[c, np.invert(np.isnan(all[c]))]
            if len(np.unique(nnanclasses)) < 2:
                auroc[c] = np.nan
                pvals[c] = np.nan
            else:
                auroc[c] = metrics.roc_auc_score(nnanclasses, nnanall)
                pvals[c] = stats.ranksums(hits[c, np.invert(np.isnan(hits[c, :]))],
                                          miss[c, np.invert(np.isnan(miss[c, :]))]).pvalue

        return (auroc, pvals, nhits, nmiss)

    def auroc_falsealarm_correctreject(self, data, trange_s=(0, 2)):
        """
        Find the cell-by-cell response to hits vs misses
        :param pars:
        :return: vector of hits - misses/mean
        """

        csargs = {
            'start-s': trange_s[0],
            'end-s': trange_s[1],
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1,
        }

        cr = None
        fa = None

        for r in data['train']:
            t2p = self.trace2p(r)

            # Of shape ncells, frames, nstimuli/onsets.
            resp = t2p.cstraces('minus', csargs)

            if np.shape(resp)[2] > 0:
                errcodes = t2p.errors('minus')
                succodes = np.invert(errcodes)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    resp = np.nanmean(resp, axis=1)
                if cr is None:
                    cr = resp[:, succodes]
                    fa = resp[:, errcodes]
                else:
                    cr = np.concatenate([cr, resp[:, succodes]], axis=1)
                    fa = np.concatenate([fa, resp[:, errcodes]], axis=1)

        ncr = np.shape(cr)[1]
        nfa = np.shape(fa)[1]
        ncell = np.shape(cr)[0]
        all = np.concatenate([fa, cr], axis=1)
        classes = np.array([0]*nfa + [1]*ncr)

        # Make sure that this is possible to compute
        if ncr == 0 or nfa == 0: return ([], [], -1, -1)

        # Get the auROC for each cell
        auroc = np.zeros(ncell)
        pvals = np.zeros(ncell)
        for c in range(ncell):
            nnanclasses = classes[np.invert(np.isnan(all[c]))]
            nnanall = all[c, np.invert(np.isnan(all[c]))]

            if len(np.unique(nnanclasses)) == 2:
                auroc[c] = metrics.roc_auc_score(nnanclasses, nnanall)
                pvals[c] = stats.ranksums(fa[c, :], cr[c, :]).pvalue
            else:
                auroc[c] = np.nan
                pvals[c] = np.nan

        return (auroc, pvals, nfa, ncr)