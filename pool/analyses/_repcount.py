import numpy as np

from . import base
from pool import config


class Repcount(base.AnalysisBase):
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
            if self.analysis('good-%s'%cs):
                (self.out['reppair_%s'%cs],
                 self.out['repcount_%s'%cs]) = self.repev(data, cs, 0.1)



                self.out['reppair-diff-0.1-%s'%cs] = (self.out['reppair-0.1-%s'%cs] -
                                                      self.out['unreppair-0.1-%s'%cs])

                if np.any(self.out['reppair-0.1-%s'%cs] + self.out['unreppair-0.1-%s'%cs] > 0):
                    self.out['reppairfrac-0.1-%s'%cs] = self.out['reppair-0.1-%s'%cs]/(
                            self.out['reppair-0.1-%s'%cs] + self.out['unreppair-0.1-%s'%cs])

                self.out['reppair-0.2-%s'%cs], _, _ = self.repev(data, cs, 0.2)
                self.out['reppair-d3-0.1-%s'%cs], _, _ = self.repev(data, cs, 0.1, 0.3)

                if 'hungry' in data and len(data['hungry']) > 0:
                    (self.out['reppair-hungry-0.1-%s'%cs],
                     self.out['repcount-hungry-0.1-%s'%cs],
                     unpair) = self.repev(data, cs, 0.1, dtype='hungry')
                    if np.any(
                            self.out['reppair-hungry-0.1-%s'%cs] + unpair > 0):
                        self.out['reppairfrac-hungry-0.1-%s'%cs] = self.out['reppair-hungry-0.1-%s'%cs]/(
                                self.out['reppair-hungry-0.1-%s'%cs] + unpair)

        return out

    requires = ['classifier']
    sets = ['repcount_%s' % cs for cs in config.stimuli()] + \
           ['repcount_hungry_%s' % cs for cs in config.stimuli()] + \
           ['repcount_iti_%s' % cs for cs in config.stimuli()] + \
           ['repcount_comb_%s' % cs for cs in config.stimuli()] + \
           ['repcount_pair_%s' % cs for cs in config.stimuli()] + \
           ['repcount_pair_hungry_%s' % cs for cs in config.stimuli()] + \
           ['repcount_pair_iti_%s' % cs for cs in config.stimuli()] + \
           ['repcount_pair_comb_%s' % cs for cs in config.stimuli()]
    across = 'day'
    updated = '180802'

    # ================================================================================== #
    # ANYTHING YOU NEED

    def repev(self, runs, cs, thresh=0.1, decthresh=0.2, trange=(-2, 3), dtype='sated'):
        """
        Find all replay events
        :param cs:
        :return:
        """

        pairrep = None
        counts = None
        unpair = None
        for run in data[dtype]:
            evs = outfns.emptynone(self.analysis('event-peaks-run%i-%.2f-%s' % (run, thresh, cs)))

            t2p = self.trace2p(run)
            trs = t2p.trace('deconvolved')

            ncells = np.shape(trs)[0]
            if pairrep is None:
                pairrep = np.zeros((ncells, ncells))
                unpair = np.zeros((ncells, ncells))
                counts = np.zeros(ncells)

            for ev in evs:
                if -1*trange[0] < ev < np.shape(trs)[1] - trange[1]:
                    act = np.nanmax(trs[:, ev+trange[0]:ev+trange[1]], axis=1)
                    act = act > decthresh

                    actout = np.zeros((len(act), len(act)), dtype=np.bool)  # pairs of cells that overlapped
                    for i in range(len(act)):  # iterate over cells
                        actout[i, :] = np.bitwise_and(act[i], act)

                    pairrep += actout.astype(np.float64)
                    counts += act.astype(np.float64)
                    unpair += np.invert(actout).astype(np.float64)

        return pairrep, counts, unpair