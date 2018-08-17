# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np


class CSBias(object):
    def __init__(self, data):
        self.out = {}

        vdrive = np.bitwise_or(self.analysis('visually-driven-plus') > 50,
                               np.bitwise_or(self.analysis('visually-driven-neutral') > 50,
                                             self.analysis('visually-driven-minus') > 50))
        vdrive30 = np.bitwise_or(self.analysis('visually-driven-plus') > 30,
                                 np.bitwise_or(self.analysis('visually-driven-neutral') > 30,
                                               self.analysis('visually-driven-minus') > 30))

        stim = None
        csstim = {}
        for cs in ['plus', 'neutral', 'minus']:
            csstim[cs] = self.analysis('stimulus-dff-all-0-2-%s' % cs)
            csstim[cs][csstim[cs] < 0] = 0

            nanzeroed = np.copy(csstim[cs])
            nanzeroed[np.isnan(nanzeroed)] = 0
            if stim is None:
                stim = nanzeroed
            else:
                stim += nanzeroed

        stim[stim == 0] = np.nan
        for cs in ['plus', 'neutral', 'minus']:
            self.out['cs-bias-%s' % cs] = csstim[cs]/stim
            self.out['cs-bias-%s' % cs][np.invert(vdrive)] = np.nan

            self.out['cs-bias-30-%s' % cs] = csstim[cs]/stim
            self.out['cs-bias-30-%s' % cs][np.invert(vdrive30)] = np.nan

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['cs-bias-%s' % cs, 'cs-bias-30-%s' % cs] for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180220'
    depends_on = ['visually-driven-%s' % cs for cs in ['plus', 'neutral', 'minus']] + \
        ['stimulus-dff-all-0-2-%s' % cs for cs in ['plus', 'neutral', 'minus']]

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

