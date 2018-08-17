# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from itertools import chain
import numpy as np
import warnings

class HungrySated(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #   classifier: whether a classifier will be required or not

    requires = []
    sets = [['n-hungry-%s'%cs,
                'n-sated-%s'%cs,
                'first-lick-median-%s'%cs,
                'hunger-modulation-%s'%cs,
                'hungry-sated-auroc-%s'%cs,
                'hungry-sated-auroc-p-%s'%cs
            ] for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '170612'

    # def trace2p(self, run):
    #   """
    #   Return trace2p file, automatically injected
    #   :param run: run number, int
    #   :return: trace2p instance
    #   """

    # def classifier(self, run, randomize=''):
    #   """
    #   Return classifier (forced to be created if it doesn't exist), automatically injected
    #   :param run: run number, int
    #   :param randomize: randomization type, optional
    #   :return:
    #   """

    # pars = {}  # dict of parameters, automatically injected

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED

    def hungry(self, data):
        """
        Get the hungry values and account for the median lick time.
        :param data: 
        :return: 
        """

        # Set up the arguments to pass to cstraces
        args = {'start-s': 0, 'end-s': 2, 'trace-type': 'dff', 'cutoff-before-lick-ms': 100, 'baseline': (-1, 0)}

        # Iterate through
        flicks = {}
        vals = {}
        for r in data['train']:
            t2p = self.trace2p(r)
            for cs in ['plus', 'neutral', 'minus']:
                trs = t2p.cstraces(cs, args)
                fls = t2p.firstlick(cs).tolist()
                # ncells, frames, nstimuli/onsets.
                if cs == 'plus':
                    pavs = t2p.cstraces('pavlovian', args)
                    if len(pavs) > 0:
                        trs = np.concatenate([trs, pavs], axis=2)
                        fls.extend(t2p.firstlick('pavlovian').tolist())

                if cs not in flicks:
                    flicks[cs] = fls
                    vals[cs] = trs
                else:
                    flicks[cs].extend(t2p.firstlick(cs).tolist())
                    vals[cs] = np.concatenate([vals[cs], trs], axis=2)

        medlick = {}
        for cs in flicks:
            flicks[cs] = np.array(flicks[cs])
            flicks[cs][np.isnan(flicks[cs])] = int(round(t2p.framerate*2 + 0.5))
            medlick[cs] = int(np.median(flicks[cs]))

            vals[cs][:, medlick[cs]+1:, :] = np.nan
            with warnings.catch_warnings():
                # We're ignoring warnings where the whole cell is nan
                warnings.simplefilter('ignore', category=RuntimeWarning)
                vals[cs] = np.nanmean(vals[cs], axis=1)
            # vals[cs] = np.nanmean(vals[cs], axis=1)

        return vals, medlick

    def sated(self, data, medlick):
        """
        Get the hungry values and account for the median lick time.
        :param data: 
        :return: 
        """

        # Set up the arguments to pass to cstraces
        args = {'start-s': 0, 'end-s': 2, 'trace-type': 'dff', 'cutoff-before-lick-ms': 100, 'baseline': (-1, 0)}

        # Get sated stimulus days
        rs = []
        if 'sated-stim' in data: rs.extend(data['sated-stim'])
        for r in data['sated']:
            if r > 8:
                rs.append(r)

        # Iterate through
        vals = {}
        for r in rs:
            t2p = self.trace2p(r)
            for cs in ['plus', 'neutral', 'minus']:
                trs = t2p.cstraces(cs, args)
                # ncells, frames, nstimuli/onsets.
                if cs == 'plus':
                    pavs = t2p.cstraces('pavlovian', args)
                    if len(pavs) > 0:
                        trs = np.concatenate([trs, pavs], axis=2)

                if cs not in vals:
                    vals[cs] = trs
                else:
                    vals[cs] = np.concatenate([vals[cs], trs], axis=2)

        for cs in vals:
            vals[cs][:, medlick[cs]+1:, :] = np.nan
            with warnings.catch_warnings():
                # We're ignoring warnings where the whole cell is nan
                warnings.simplefilter('ignore', category=RuntimeWarning)
                vals[cs] = np.nanmean(vals[cs], axis=1)
            # vals[cs] = np.nanmean(vals[cs], axis=1)

        return vals

    def auroc_p(self, group1, group2):
        from scipy import stats
        from sklearn import metrics

        ng1 = np.shape(group1)[1]
        ng2 = np.shape(group2)[1]
        ncell = np.shape(group1)[0]

        all = np.concatenate([group1, group2], axis=1)
        classes = np.array([0]*ng1 + [1]*ng2)

        # Make sure that this is possible to compute
        if ng1 == 0 or ng2 == 0: return [np.nan, np.nan]

        # Get the auROC for each cell
        auroc = np.zeros(ncell)
        pvals = np.zeros(ncell)
        for c in range(ncell):
            # Find all of the time points that are nan and exclude them
            nnanclasses = classes[np.invert(np.isnan(all[c]))]
            nnanall = all[c, np.invert(np.isnan(all[c]))]
            if len(np.unique(nnanclasses)) < 2:
                auroc[c] = np.nan
                pvals[c] = np.nan
            else:
                auroc[c] = metrics.roc_auc_score(nnanclasses, nnanall)
                pvals[c] = stats.ranksums(group1[c, np.invert(np.isnan(group1[c, :]))],
                                          group2[c, np.invert(np.isnan(group2[c, :]))]).pvalue

        return (auroc, pvals)

    def analyze(self, data):
        """
        Return a vector of the replay-weighted activity for a stimulus cs, a classifier gm, a trace2p file t2p
        :param pars:
        :param gm:
        :param t2p:
        :param cs:
        :return:
        """

        h, lick = self.hungry(data)
        s = self.sated(data, lick)

        for cs in h:
            self.out['n-hungry-%s'%(cs)] = np.shape(h[cs])[1]
            self.out['n-sated-%s'%(cs)] = 0
            self.out['first-lick-median-%s'%(cs)] = np.nan
            self.out['hunger-modulation-%s'%(cs)] = np.nan
            self.out['hungry-sated-auroc-%s'%(cs)], self.out['hungry-sated-auroc-p-%s'%(cs)] = np.nan, np.nan

            if cs in s and np.shape(s[cs])[1] > 0:
                self.out['n-sated-%s'%(cs)] = np.shape(s[cs])[1]

                if len(h[cs]) > 0 and len(s[cs]) > 0:
                    mnh = np.nanmean(h[cs], axis=1)
                    mns = np.nanmean(s[cs], axis=1)
                    self.out['first-lick-median-%s' % (cs)] = lick[cs]
                    self.out['hunger-modulation-%s' % (cs)] = (mnh - mns)/(mnh + mns)
                    self.out['hungry-sated-auroc-%s' % (cs)], self.out['hungry-sated-auroc-p-%s' % (cs)] = self.auroc_p(s[cs], h[cs])