# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from copy import deepcopy
import numpy as np

class PeakTime(object):
    def __init__(self, data):
        self.out = {}

        for endsec in ['2', '2.5', '4']:
            for ttype in ['', '-correct', '-false']:
                res = self.getcom(data, ttype, float(endsec))
                for cs in res:
                    self.out['peak-com%s-0-%s-%s' % (ttype, endsec, cs)] = res[cs]

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [[
            'peak-com%s-0-4-%s'%(ttype, cs),
            'peak-com%s-0-2-%s'%(ttype, cs),
            'peak-com%s-0-2.5-%s'%(ttype, cs),
        ] for cs in ['plus', 'neutral', 'minus'] for ttype in ['', '-correct', '-false']]
    across = 'day'
    updated = '171112'

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

    def get_confidence(self, trs, frac=0.2, rep=100):
        """
        Get the confidence
        :param trs: traces of shape ncells, nframes, nonsets
        :param frac: fraction of traces from which we determine the mean
        :param rep: number of randomized repetitions
        :return: confidence vector, per cell
        """

        ntrials = np.shape(trs)[2]
        timetrials = np.zeros((np.shape(trs)[0], rep), dtype=np.float32)
        for i in range(rep):
            subtrs = trs[:, :, np.random.randint(0, ntrials, int(frac*ntrials))]
            timetrials[:, i] = np.argmax(np.nanmean(subtrs, axis=2), axis=1)
        return np.nanstd(timetrials, axis=1)

    def getcom(self, data, errors, ends):
        """
        Get all stimuli and take means across different time periods.
        :param data:
        :return:
        """

        # import matplotlib.pyplot as plt
        # import time

        argsf = {
            'start-s': 0,
            'end-s': ends,
            'trace-type': 'dff',
            'cutoff-before-lick-ms': -1,
            'error-trials': -1 if errors == '' else 0 if errors == '-correct' else 1,
            'baseline': (-1, 0),
        }

        # Go through the added stimuli and add all onsets
        results = {}
        for cs in ['plus', 'neutral', 'minus']:
            trs = []
            for r in data['train']:
                argsb = deepcopy(argsf)

                if len(trs) == 0:
                    trs = self.trace2p(r).cstraces(cs, argsb)  # ncells, frames, nstimuli/onsets
                else:
                    trs = np.concatenate([trs, self.trace2p(r).cstraces(cs, argsb)], axis=2)

                if cs == 'plus':
                    trs = np.concatenate([trs, self.trace2p(r).cstraces('pavlovian', argsb)], axis=2)

            if len(trs) == 0 or np.shape(trs)[2] == 0:
                results[cs] = []
            else:
                nframes = np.shape(trs)[1]
                frames = np.arange(nframes)

                trs = np.nanmean(trs, axis=2)
                peaks = np.zeros(np.shape(trs)[0])
                peaks[:] = np.nan
                vdrive = self.analysis('visually-driven-%s'%cs)
                trs[np.invert(np.isfinite(trs))] = 0
                trs[trs < 0.00001] = 0.00001

                for c in range(np.shape(trs)[0]):
                    peaks[c] = np.average(frames, weights=trs[c, :])
                    # plt.plot(frames, trs[c, :])
                    # plt.plot([peaks[c], peaks[c]], [0, np.nanmax(trs[c, :])])
                    # print peaks[c], vdrive[c]
                    # plt.show()
                    results[cs] = peaks/self.trace2p(r).framerate
        return results
