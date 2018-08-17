# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns


class GLMCosdist(object):
    def __init__(self, data):
        self.out = {}
        for cs in ['plus', 'minus']:
            framerate = self.analysis('framerate-run%i' % data['sated'][0])
            keep = np.invert(np.bitwise_or(self.analysis('activity-outliers'),
                                                self.analysis('activity-sated-outliers')))

            proto = outfns.protovectors(data['mouse'], data['date'], keep=keep, hz=framerate)
            self.out['glm-uscsdist-%s' % cs] = outfns.uscsdist(proto, cs)
            self.out['lsq-uscsdist-%s' % cs] = outfns.uscsdist(proto, cs, True)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['glm-uscsdist-%s' % cs, 'lsq-uscsdist-%s' % cs] for cs in ['plus', 'minus']]
    across = 'day'
    updated = '180203'

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

