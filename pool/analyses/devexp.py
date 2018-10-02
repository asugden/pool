# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from . import base
import flow.glm as glm


class Stim(base.AnalysisBase):
    def run(self, mouse, date, training, running, sated, hungry):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        mouse : str
            mouse name
        date : str
            current date
        training : list of ints
            list of training run numbers as integers
        running : list of ints
            list of running-only run numbers as integers
        sated : list of ints
            list of sated spontaneous run numbers as integers
        hungry : list of ints
            list of hungry spontaneous run numbers as integers

        Returns
        -------
        dict
            All of the output values

        """

        out = {}

        beh = glm.glm(mouse, date)
        expl = beh.explained()
        for cellgroup in expl:
            out['glm-devexp-%s' % cellgroup] = expl[cellgroup]

        for cellgroup in ['plus', 'neutral', 'minus', 'ensure', 'quinine', 'lick']:
            if 'glm-devexp-%s'%cellgroup not in out:
                out['glm-devexp-%s'%cellgroup] = np.nan

        return out

    requires = ['']
    sets = ['glm-devexp-plus',
            'glm-devexp-neutral',
            'glm-devexp-minus',
            'glm-devexp-ensure',
            'glm-devexp-quinine',
            'glm-devexp-lick']
    across = 'day'
    updated = '180911'