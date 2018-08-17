# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

from flow import outfns, paths


class GLMLabels(object):
    def __init__(self, data):
        self.out = {}
        lbls = self.labels(data)

        for sname in self.sets:
            if '-of-' not in sname:
                self.out[sname] = float(np.sum(lbls[sname.replace('glmfrac-', '')]))/\
                                  len(lbls[sname.replace('glmfrac-', '')])

        self.out['glmfrac-ensure-vdrive-plus-of-ensure'] = outfns.nandivide(self.out['glmfrac-ensure-vdrive-plus'],
                                                                            self.out['glmfrac-ensure'])
        self.out['glmfrac-ensure-vdrive-plus-nolick-of-ensure'] = outfns.nandivide(
            self.out['glmfrac-ensure-vdrive-plus-nolick'], self.out['glmfrac-ensure-nolick'])

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [
            'glmfrac-ensure',
            'glmfrac-ensure-only',
            'glmfrac-ensure-nolick',
            'glmfrac-quinine',
            'glmfrac-quinine-only',
            'glmfrac-quinine-nolick',
            'glmfrac-plus',
            'glmfrac-plus-only',
            'glmfrac-plus-nolick',
            'glmfrac-plus-multiplexed-nolick',
            'glmfrac-minus',
            'glmfrac-neutral',
            'glmfrac-undefined',
            'glmfrac-lick',
            'glmfrac-plus-ensure',
            'glmfrac-ensure-vdrive-plus',
            'glmfrac-ensure-vdrive-neutral',
            'glmfrac-ensure-vdrive-minus',
            'glmfrac-ensure-vdrive-plus-nolick',
            'glmfrac-ensure-vdrive-neutral-nolick',
            'glmfrac-ensure-vdrive-minus-nolick',
            'glmfrac-minus-quinine',
            'glmfrac-plus-vdrive',
            'glmfrac-minus-vdrive',
            'glmfrac-plus-vdrive-nolick',
            'glmfrac-ensure-vdrive-plus-of-ensure',
            'glmfrac-ensure-vdrive-plus-nolick-of-ensure',
    ]
    across = 'day'
    updated = '180219'

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def labels(self, data, minpred=0.01, minfrac=0.05):
        # Get cell groups and the sufficiency of each deviance explained

        gdata = paths.getglm(data['mouse'], data['date'])
        odict, multiplexed = outfns.labelglm(gdata, minpred, minfrac)
        odict['undefined'] = multiplexed == 0

        odict['multiplexed'] = multiplexed > 1
        categories = ['plus', 'neutral', 'minus', 'ensure', 'quinine', 'lick']
        for cat in categories:
            odict['%s-only'%cat] = np.bitwise_and(odict[cat], np.invert(odict['multiplexed']))
            odict['%s-multiplexed'%cat] = np.bitwise_and(odict[cat], odict['multiplexed'])

        odict['plus-ensure'] = np.bitwise_and(odict['plus'], odict['ensure'])
        odict['minus-quinine'] = np.bitwise_and(odict['minus'], odict['quinine'])
        odict['ensure-vdrive-plus'] = np.bitwise_and(self.analysis('visually-driven-plus') > 50, odict['ensure'])
        odict['ensure-vdrive-neutral'] = np.bitwise_and(self.analysis('visually-driven-neutral') > 50, odict['ensure'])
        odict['ensure-vdrive-minus'] = np.bitwise_and(self.analysis('visually-driven-minus') > 50, odict['ensure'])
        odict['plus-vdrive'] = self.analysis('visually-driven-plus') > 50
        odict['minus-vdrive'] = self.analysis('visually-driven-minus') > 50

        for ctype in ['ensure', 'plus', 'undefined', 'quinine', 'plus-multiplexed', 'ensure-vdrive-plus',
                      'ensure-vdrive-neutral', 'ensure-vdrive-minus', 'plus-vdrive']:
            odict['%s-nolick' % ctype] = np.copy(odict[ctype])
            odict['%s-nolick' % ctype][odict['lick']] = False

        return odict