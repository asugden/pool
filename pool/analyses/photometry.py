# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import metadata

class Photometry(object):
    def __init__(self, data):
        self.out = {key:np.nan for key in self.sets}

        md = metadata.data(data['mouse'], data['date'])
        if 'photometry' in md and 'nacc' in md['photometry']:
            self.photstim(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = []
    sets = ['photometry-%s-%s' % (cs, cor) for cs in ['plus', 'neutral', 'minus'] for cor in ['correct', 'miss']]
    across = 'day'
    updated = '171230'

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

    def photstim(self, data):
        tbaseline = [-1, 0]
        trange = [0, 6]

        vals = {}
        for r in data['train']:
            t2p = self.trace2p(r)
            if t2p.hasphotometry():
                phot = t2p.photometry(tracetype='dff')

                frange = [int(round(t2p.framerate*trange[0])), int(round(t2p.framerate*trange[1]))]
                fbaseline = [int(round(t2p.framerate*tbaseline[0])), int(round(t2p.framerate*tbaseline[1]))]

                for cs in ['plus', 'neutral', 'minus']:
                    for trialtype in range(2):
                        cor = 'correct' if trialtype == 0 else 'miss'
                        onsets = t2p.csonsets(cs, errortrials=trialtype)

                        if 'photometry-%s-%s' % (cs, cor) not in vals:
                            vals['photometry-%s-%s' % (cs, cor)] = []

                        for ons in onsets:
                            if ons + fbaseline[0] >= 0 and ons + frange[1] < len(phot):
                                ptr = phot[ons + frange[0]:ons + frange[1]]
                                ptr -= np.nanmean(phot[ons + fbaseline[0]:ons + fbaseline[1]])
                                vals['photometry-%s-%s' % (cs, cor)].append(np.nanmean(ptr))

        for key in vals:
            self.out[key] = np.nanmean(vals[key])
