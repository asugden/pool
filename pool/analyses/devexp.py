import numpy as np

from . import base


class Devexp(base.AnalysisBase):
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

        beh = date.glm()
        expl = beh.explained()
        for cellgroup in expl:
            out['devexp_%s' % cellgroup] = expl[cellgroup]

        return out

    requires = ['']
    sets = ['devexp_plus',
            'devexp_neutral',
            'devexp_minus',
            'devexp_ensure',
            'devexp_quinine',
            'devexp_lick',
            'devexp_total']
    across = 'day'
    updated = '180911'