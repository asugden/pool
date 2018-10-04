import numpy as np

from . import base
from pool import config


class Repcount(base.AnalysisBase):
    requires = ['classifier']
    sets = ['repcount_%s'%cs for cs in config.stimuli()] + \
           ['repcount_hungry_%s'%cs for cs in config.stimuli()] + \
           ['repcount_iti_%s'%cs for cs in config.stimuli()] + \
           ['repcount_comb_%s'%cs for cs in config.stimuli()] + \
           ['repcount_pair_%s'%cs for cs in config.stimuli()] + \
           ['repcount_pair_hungry_%s'%cs for cs in config.stimuli()] + \
           ['repcount_pair_iti_%s'%cs for cs in config.stimuli()] + \
           ['repcount_pair_comb_%s'%cs for cs in config.stimuli()]
    across = 'day'
    updated = '181006'

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
            if self.analysis('good_%s'%cs):
                out['repcount_%s'%cs], out['repcount_pair_%s'%cs] = \
                    self.event_counts(date.runs('spontaneous', 'sated'), cs, 0.1)

                out['repcount_hungry_%s'%cs], out['repcount_pair_hungry_%s'%cs] = \
                    self.event_counts(date.runs('spontaneous', 'hungry'), cs, 0.1)

                out['repcount_iti_%s'%cs], out['repcount_pair_iti_%s'%cs] = \
                    self.event_counts(date.runs('training'), cs, 0.1)

                out['repcount_comb_%s'%cs], out['repcount_pair_comb_%s'%cs] = \
                    self.event_counts(date.runs(), cs, 0.1)

        return out

    def event_counts(self, runs, cs, deconvolved_threshold=0.2, trange=(-2, 3)):
        """
        Find all replay events
        :param cs:
        :return:
        """

        count_pair = None
        count = None
        for run in runs:
            evs = self.analysis('repevent_%s' % cs, run)
            t2p = run.trace2p()

            if count_pair is None:
                count_pair = np.zeros((t2p.ncells, t2p.ncells))
                count = np.zeros(t2p.ncells)

            if evs is not None:
                trs = t2p.trace('deconvolved')

                for ev in evs:
                    if -1*trange[0] < ev < np.shape(trs)[1] - trange[1]:
                        act = np.nanmax(trs[:, ev+trange[0]:ev+trange[1]], axis=1)
                        act = act > deconvolved_threshold

                        actout = np.zeros((len(act), len(act)), dtype=np.bool)  # pairs of cells that overlapped
                        for i in range(len(act)):  # iterate over cells
                            actout[i, :] = np.bitwise_and(act[i], act)

                        count_pair += actout.astype(np.float64)
                        count += act.astype(np.float64)

        return count, count_pair
