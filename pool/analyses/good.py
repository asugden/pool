import numpy as np

from . import base
from pool import config


class Good(base.AnalysisBase):
    requires = ['']
    sets = ['good_%s'%cs for cs in config.stimuli()]
    across = 'day'
    updated = '181003'

    def run(self, date):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        date : Date object

        Returns
        -------
        dict
            All of the output values

        """

        out = self.nanoutput()

        for cs in config.stimuli():
            stimresp = self.analysis('stim_dff_alltrials_%s' % cs)
            dff_active = np.sum(stimresp > 0.025)/float(len(stimresp))
            out['good_%s' % cs] = dff_active > 0.05

        return out
