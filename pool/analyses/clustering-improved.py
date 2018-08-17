# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get
import numpy as np

from flow import labels

from .base import AnalysisBase
from .. import netcom


class BetterClustering(AnalysisBase):
    def _run_analyses(self):
        out = {}
        nx = netcom.graph(self.andb, 'hofer', vdrive=50)
        ing, outg = nx.groupconnectivity()
        out['cluster-within-hofer'] = np.nanmean(ing)
        out['cluster-between-hofer'] = np.nanmean(outg)

        lbls = labels.categorize(self._data['mouse'], self._data['date'])
        nx = netcom.graph(self.andb, 'plus', vdrive=50)
        nx.label(lbls)
        ing, outg = nx.groupconnectivity()
        iconnrew, oconnrew, iconnnon, oconnnon = nx.labelconnectivity('ensure')
        out['cluster-within-plus'] = np.nanmean(ing)
        out['cluster-between-plus'] = np.nanmean(outg)
        out['cluster-within-reward'] = np.nanmean(iconnrew)
        out['cluster-between-reward'] = np.nanmean(oconnrew)
        out['cluster-within-nonreward'] = np.nanmean(iconnnon)
        out['cluster-between-nonreward'] = np.nanmean(oconnnon)

        conn = nx.connectivity()
        clust, _ = nx.clusterlabel('ensure', exclusive=False, count=1)
        react = self.analysis('replay-count-0.1-plus')
        out['total-connectivity'] = 0.0
        out['total-connectivity-ensure'] = 0.0
        out['total-connectivity-reward'] = 0.0

        for c in range(len(conn)):
            if np.isfinite(conn[c]):
                out['total-connectivity'] += conn[c]

                if lbls[c] == 'ensure' and react is not None and react[c]:
                    out['total-connectivity-ensure'] += conn[c]

                if (lbls[c] == 'ensure' or clust[c]) and react is not None and react[c]:
                    out['total-connectivity-reward'] += conn[c]

        if out['total-connectivity'] == 0:
            out['frac-total-connectivity-ensure'] = np.nan
            out['frac-total-connectivity-reward'] = np.nan
        else:
            out['frac-total-connectivity-ensure'] = out['total-connectivity-ensure']/out['total-connectivity']
            out['frac-total-connectivity-reward'] = out['total-connectivity-reward']/out['total-connectivity']

        out['n-cells'] = len(conn)

        ens, rew, tot = self.totnoisecorr(lbls, clust)

        out['total-noisecorr-ensure'] = ens
        out['total-noisecorr-reward'] = rew
        out['total-noisecorr'] = tot

        if out['total-noisecorr'] == 0:
            out['frac-total-noisecorr-ensure'] = np.nan
            out['frac-total-noisecorr-reward'] = np.nan
        else:
            out['frac-total-noisecorr-ensure'] = out['total-noisecorr-ensure']/out['total-noisecorr']
            out['frac-total-noisecorr-reward'] = out['total-noisecorr-reward']/out['total-noisecorr']

        return out

    def totnoisecorr(self, lbls, clust):
        """
        Get total noise correlations for ensure and reward-cluster cells
        :return:
        """

        ncs = self.analysis('noise-correlation-plus')
        limits = self.analysis('visually-driven-plus') > 50

        tot = 0.0
        ens = 0.0
        rew = 0.0
        for c1 in range(len(clust)):
            for c2 in range(c1+1, len(clust)):
                if limits[c1] and limits[c2]:
                    tot += ncs[c1, c2]

                    if lbls[c1] == 'ensure':
                        ens += ncs[c1, c2]

                    if lbls[c1] == 'ensure' or clust[c1]:
                        rew += ncs[c1, c2]

        return ens, rew, tot





    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['cluster-within-%s' % cs,
             'cluster-between-%s' % cs] for cs in ['hofer', 'plus', 'reward', 'nonreward']] + \
            ['total-connectivity', 'total-connectivity-ensure', 'frac-total-connectivity-ensure', 'n-cells',
             'total-connectivity-reward', 'frac-total-connectivity-reward', 'total-noisecorr-ensure',
             'frac-total-noisecorr-ensure', 'total-noisecorr-reward', 'frac-total-noisecorr-reward']
    across = 'day'
    updated = '180508'
