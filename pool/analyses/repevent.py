from copy import deepcopy
import numpy as np

from . import base
from pool import config


class Repevent(base.AnalysisBase):
    requires = ['classifier']
    sets = ['repevent_%s' % cs for cs in config.stimuli()] + \
           ['repevent_inactive_frames']
    across = 'run'
    updated = '181005'

    def run(self, run):
        """
        Run all analyses and returns results in a dictionary.

        Parameters
        ----------
        run : Run object

        Returns
        -------
        dict
            All of the output values

        """

        out = self.nanoutput()
        t2p = run.trace2p()
        trs = t2p.trace('deconvolved')

        if run.run_type != 'training':
            c2p = run.classify2p()
            mask = t2p.inactivity()
        else:
            c2p = self.training_classifier(run)

            # Take the times that were during stimuli or during inactivity and block them
            mask = t2p.trialmask(padpre=0.1, padpost=0.5)
            mask = np.bitwise_or(mask, t2p.inactivity(nostim=False))
            mask = np.invert(mask)

        out['repevent_inactive_frames'] = np.sum(mask)

        for cs in config.stimuli():
            if self.analysis('good_%s'%cs):
                out['repevent_%s' % cs] = c2p.events(
                    cs, 0.1, trs, mask=mask, xmask=True,
                    max=2, downfor=2, maxlen=-1)

        return out

    def training_classifier(self, run):
        """
        Get a training classifier instance

        Parameters
        ----------
        run : Run object
            The run to be classified

        Returns
        -------
        Classify2p object

        """

        # Get a classifier instance
        parameters = deepcopy(self.pars)
        parameters['training-runs'] = [r.run for r in run.parent.runs('training') if r.run != run.run]
        parameters['comparison-run'] = run.run

        gm = run.classify2p(parameters)

        return gm