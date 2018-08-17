# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns

from .base import AnalysisBase


class DeconActivity(AnalysisBase):
    def _run_analyses(self):
        out = {}

        keep = np.invert(np.bitwise_or(self.analysis('activity-outliers'), self.analysis('activity-sated-outliers')))

        act = []
        for run in self._data['sated']:
            t2p = self.trace2p(run)
            mask = t2p.inactivity()
            fmin = t2p.lastonset()
            mask[:fmin] = False
            trs = t2p.trace('deconvolved')[keep, :]
            trs = trs[:, mask]

            if len(act) == 0:
                act = trs
            else:
                act = np.concatenate([act, trs], axis=1)

        out['mean-deconvolved-sated-activity'] = np.nanmean(act)

        return out

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    requires = ['']
    sets = ['mean-deconvolved-sated-activity']
    across = 'day'
    updated = '180508'
