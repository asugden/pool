# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns

class OriReplayFreq(object):
    def __init__(self, data):
        self.out = {}

        mousedict = {
            # pre 'CB173': {'plus': 0, 'neutral': 135, 'minus': 270},
            'CB173': {'plus': 135, 'neutral': 0, 'minus': 270},
            'AS20': {'plus': 0, 'neutral': 135, 'minus': 270},
            'OA32': {'plus': 0, 'neutral': 135, 'minus': 270},
            'OA34': {'plus': 135, 'neutral': 270, 'minus': 0},
            'OA36': {'plus': 270, 'neutral': 0, 'minus': 135},
            'OA37': {'plus': 270, 'neutral': 0, 'minus': 135},
            'OA38': {'plus': 0, 'neutral': 270, 'minus': 135},
            'AS41': {'plus': 0, 'neutral': 135, 'minus': 270},
        }

        for cs in [0, 135, 270]:
            corrcs = 'plus' if cs == 0 else 'neutral' if cs == 135 else 'minus'

            if data['mouse'] not in mousedict:
                print 'WARNING WILL ROBINSON'
                self.out['replay-freq-ori-0.1-%s'%(cs)] = np.nan
                self.out['replay-freq-ori-hungry-0.1-%s'%(cs)] = np.nan
            else:
                for cs2 in ['plus', 'neutral', 'minus']:
                    if mousedict[data['mouse']][cs2] == cs:
                        self.out['replay-freq-ori-0.1-%s'%(corrcs)] = self.analysis('replay-freq-0.1-%s'%cs2)
                        self.out['replay-freq-ori-hungry-0.1-%s'%(corrcs)] = self.analysis('replay-freq-0.1-%s'%cs2)

        print self.out


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['replay-freq-ori-0.1-%s'%(cs), 'replay-freq-ori-hungry-0.1-%s'%(cs)]
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180521'

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
