# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

class FalsePositives(object):
    def __init__(self, data):
        self.out = {}
        self.get_false_positives_identity_removed(data)
        for rtype in ['circshift', 'identity']:
            self.get_false_positives(data, rtype)

            if rtype == 'circshift':
                self.get_qualities(rtype)

        # self.repidentityfp(data)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = ['quality-circshift-mean'] + \
           [['quality-circshift-%s' % cs,
             'n-real-events-%s' % cs,
             'false-positive-rem-%s' % cs,
             'n-randrem-events-%s' % cs]
            for cs in ['plus', 'neutral', 'minus']] + \
           [['false-positive-%s-%s' % (rand, cs), 'n-rand-events-%s-%s' % (rand, cs)]
            for rand in ['circshift', 'identity']
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180203'

    # def trace2p(self, run):
    # 	"""
    # 	Return trace2p file, automatically injected
    # 	:param run: run number, int
    # 	:return: trace2p instance
    # 	"""

    # def classifier(self, run, randomize='', n=1):
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

    def get_safe(self, rtype, threshold=0.75):
        """
        Get the classifier thresholds above which one is guaranteed to have > 0.75 true positives
        :return:
        """

        for cs in ['plus', 'neutral', 'minus']:
            if self.out['false-positive-%s-%s'%(rtype, cs)] is None: self.out['safe-%s-%s'%(rtype, cs)] = None
            else:
                self.out['safe-%s-%s'%(rtype, cs)] = 1.1
                fp = 20
                while fp > 0 and (not np.isfinite(self.out['false-positive-%s-%s'%(rtype, cs)][fp-1]) or
                                    self.out['false-positive-%s-%s'%(rtype, cs)][fp-1] > threshold):
                    self.out['safe-%s-%s'%(rtype, cs)] = float(fp-1)/20.0
                    fp -= 1

        self.out['safe-%s-max'%rtype] = max(self.out['safe-%s-plus'%rtype],
                                            self.out['safe-%s-neutral'%rtype],
                                            self.out['safe-%s-minus'%rtype])

    def get_qualities(self, rtype):
        """
        Get the quality score for each CS based on the false-positive rates of each stimulus
        :return:
        """

        mn = 0
        for cs in ['plus', 'neutral', 'minus']:
            if self.out['false-positive-%s-%s'%(rtype, cs)] is None:
                self.out['quality-%s-%s'%(rtype, cs)] = None

            else:
                # Anything below 0.5 means that everything can be explained due to randomness, so we'll set 0.5 to 0
                # fp = 2*(np.copy(self.out['false-positive-%s-%s'%(rtype, cs)][1:]) - 0.5)
                fp = np.copy(self.out['false-positive-%s-%s'%(rtype, cs)][1:])
                fp = fp[np.isfinite(fp)]

                fp[fp < 0] = 0
                fp = np.nanmean(fp)

                self.out['quality-%s-%s'%(rtype, cs)] = fp
                mn += self.out['quality-%s-%s'%(rtype, cs)]/3.0

        if mn == 0:
            self.out['quality-%s-mean'%rtype] = None
        else:
            self.out['quality-%s-mean'%rtype] = mn

    def get_false_positives(self, data, rtype):
        """
        Return the false positive histograms for each cs
        :param data:
        :return:
        """

        stims = ['plus', 'neutral', 'minus']
        nrand = 10
        real, rand = None, None

        # for cs in stims: self.out['rand-run-events-%s-%s'%(rtype, cs)] = {}
        # self.out['frames-used'] = []

        # Iterate over all spontaneous days
        for run in data['sated']:
            if np.sum(self.trace2p(run).inactivity()) > 1000:
                real = self.gethistrates([self.classifier(run)], stims, [self.trace2p(run)], real)
                rand = self.gethistrates(self.classifier(run, rtype, nrand), stims, None, rand)
                # randrun = self.gethistrates(self.classifier(run, rtype, nrand), stims, None, None)
                #
                # for cs in randrun:
                #     self.out['rand-run-events-%s-%s'%(rtype, cs)][run] = np.sum(randrun[cs][1:])

        for cs in stims:
            if real is None or rand is None:
                self.out['false-positive-%s-%s'%(rtype, cs)] = None
                self.out['n-real-events-%s'%cs] = None
                self.out['n-rand-events-%s-%s'%(rtype, cs)] = None
            else:
                self.out['n-real-events-%s'%cs] = real[cs]
                self.out['n-rand-events-%s-%s'%(rtype, cs)] = rand[cs]

                totreal = float(np.nansum(real[cs]))
                totrand = float(np.nansum(rand[cs]))
                rand[cs] *= totreal/totrand

                # Make each value sums from the value up rather than bins
                for i in range(len(real[cs]) - 1):
                    real[cs][i] = np.nansum(real[cs][i:])
                    rand[cs][i] = np.nansum(rand[cs][i:])

                # Get rid of NaNs
                with np.errstate(invalid='ignore'):
                    fprat = real[cs]/(real[cs] + rand[cs])
                    fprat[np.invert(np.isfinite(fprat))] = 1.0
                    fprat = np.clip(fprat, 0.0, 1.0)

                    self.out['false-positive-%s-%s'%(rtype, cs)] = fprat

    def get_false_positives_identity_removed(self, data):
        """
        Return the false positive histograms for each cs
        :param data:
        :return:
        """

        stims = ['plus', 'neutral', 'minus']
        nrand = 10
        real, randrem = None, None

        # Iterate over all spontaneous days
        for run in data['sated']:
            if np.sum(self.trace2p(run).inactivity()) > 1000:
                rlgm = self.classifier(run)
                t2p = self.trace2p(run)
                real = self.gethistrates([rlgm], stims, [t2p], real)
                randrem = self.getcorrrandhist(self.classifier(run, 'identity', nrand), stims, randrem, rlgm, t2p)

        for cs in stims:
            if real is None or randrem is None:
                self.out['false-positive-rem-%s'% cs] = None
                self.out['n-randrem-events-%s'%(cs)] = None
            else:
                self.out['n-randrem-events-%s'%(cs)] = randrem[cs]

                totreal = float(np.nansum(real[cs]))
                totrandrem = float(np.nansum(randrem[cs]))
                randrem[cs] *= totreal/totrandrem

                # Make each value sums from the value up rather than bins
                for i in range(len(real[cs]) - 1):
                    real[cs][i] = np.nansum(real[cs][i:])
                    randrem[cs][i] = np.nansum(randrem[cs][i:])

                # Get rid of NaNs
                with np.errstate(invalid='ignore'):
                    fprat = real[cs]/(real[cs] + randrem[cs])
                    fprat[np.invert(np.isfinite(fprat))] = 1.0
                    fprat = np.clip(fprat, 0.0, 1.0)

                    self.out['false-positive-rem-%s'%(cs)] = fprat

    def get_false_positives_likelihood(self, data, rtype):
        """
        Return the false positive histograms for each cs
        :param data:
        :return:
        """

        stims = ['plus', 'neutral', 'minus']
        real, rand = None, None

        # Iterate over all spontaneous days
        for run in data['sated']:
            if np.sum(self.trace2p(run).inactivity()) > 1000:
                real = self.gethistrates([self.classifier(run)], stims, [self.trace2p(run)], real, 1e-11)
                rand = self.gethistrates(self.classifier(run, rtype, 10), stims, None, rand, 1e-11)

        for cs in stims:
            if real is None or rand is None:
                self.out['false-positive-hilike-%s-%s'%(rtype, cs)] = None
                self.out['n-real-events-hilike-%s'%cs] = None
                self.out['n-rand-events-hilike-%s-%s'%(rtype, cs)] = None
            else:
                self.out['n-real-events-hilike-%s'%cs] = real[cs]
                self.out['n-rand-events-hilike-%s-%s'%(rtype, cs)] = rand[cs]

                totreal = float(np.nansum(real[cs]))
                totrand = float(np.nansum(rand[cs]))
                rand[cs] *= totreal/totrand

                # Make each value sums from the value up rather than bins
                for i in range(len(real[cs]) - 1):
                    real[cs][i] = np.nansum(real[cs][i:])
                    rand[cs][i] = np.nansum(rand[cs][i:])

                # Get rid of NaNs
                with np.errstate(invalid='ignore'):
                    fprat = real[cs]/(real[cs] + rand[cs])
                    fprat[np.invert(np.isfinite(fprat))] = 1.0
                    fprat = np.clip(fprat, 0.0, 1.0)

                    self.out['false-positive-hilike-%s-%s'%(rtype, cs)] = fprat

    def repidentityfp(self, data):
        """
        Return the false positive histograms for each cs for randomization by replay identity
        :param data:
        :return:
        """

        stims = ['plus', 'neutral', 'minus']
        rand = None

        for cs in stims: self.out['rand-run-events-repidentity-%s' % cs] = {}

        # Iterate over all spontaneous days
        for run in data['sated']:
            if np.sum(self.trace2p(run).inactivity()) > 1000:
                rand = self.getrihistrates(self.classifier(run, 'repidentity', 1), stims, rand)
                # randrun = self.getrihistrates(self.classifier(run, 'repidentity', 1), stims, None)
                # for cs in randrun:
                #     self.out['rand-run-events-repidentity-%s' % cs][run] = np.sum(randrun[cs][1:])

        for cs in stims:
            real = self.out['n-real-events-%s'%cs]

            if rand is None:
                self.out['false-positive-repidentity-%s'% cs] = None
                self.out['n-rand-events-repidentity-%s' % cs] = None
            elif real is not None:
                self.out['n-rand-events-repidentity-%s' % cs] = rand[cs]

                # Get rid of NaNs
                with np.errstate(invalid='ignore'):
                    fprat = 1.0 - rand[cs]/real
                    fprat[np.invert(np.isfinite(fprat))] = 1.0
                    fprat = np.clip(fprat, 0.0, 1.0)

                    self.out['false-positive-repidentity-%s' % cs] = fprat

    def getrihistrates(self, classifier, stimuli=['plus', 'neutral', 'minus'], appendtohist=None, bins=20):
        """
        Return the false positive histogram for replay identity classifier. If traces are
        included, their pupil maks will be applied to the output of the classifier.
        :param classifiers: list of classifier outputs
        :param traces: trace2p files for each classifier, only if not random
        :param bins: number of histogram bins
        :param appendtohist: append the results to an existing dict of histograms (for combining across days/days)
        :return: dict of histograms for each stimulus
        """

        # Append to the previous histogram, or initialize
        hist = appendtohist
        if hist is None: hist = {key: np.zeros(bins) for key in stimuli}
        threshes = np.linspace(0, 1, 20, False)

        if len(classifier['frame-match']) == 0: return hist
        elif classifier['frame-match'].ndim == 1: classifier['frame-match'] = np.array([classifier['frame-match']])

        reps = classifier['replicates']
        for key in stimuli:
            if key in classifier['codes']:
                csindex = classifier['codes'].tolist().index(key)

                for t in range(1, len(threshes)):
                    thresh = threshes[t]
                    frs = np.nonzero((classifier['frame-match'][:, 0] == csindex) &
                                     (classifier['frame-match'][:, 2] >= thresh))

                    for fr in frs[0]:
                        comp = fr*(reps + 1)
                        hist[key][t] += np.sum(classifier['results'][key][comp + 1:comp + reps + 1] >
                                               classifier['results'][key][comp])/float(reps)

        return hist

    def getcorrrandhist(self, classifiers, cses, appendtohist=None, real=None, t2p=None):
        """
        Return the false positive histogram for classifiers in classifiers. Exclude traces if random. If traces are
        included, their pupil maks will be applied to the output of the classifier.
        :param classifiers: list of classifier outputs
        :param traces: trace2p files for each classifier, only if not random
        :param bins: number of histogram bins
        :param appendtohist: append the results to an existing dict of histograms (for combining across days/days)
        :return: dict of histograms for each stimulus
        """

        bins = 20

        # Append to the previous histogram, or initialize
        hist = appendtohist
        if hist is None: hist = {key: np.zeros(bins) for key in cses}

        actmask = t2p.inactivity()
        maxmask = np.zeros(np.sum(actmask))
        for cs in cses:
            maxmask = np.nanmax([maxmask, real['results'][cs][actmask]], axis=0)

        # Iterate over listed classifiers
        for i, cl in enumerate(classifiers):
            # Random events for a single classifier

            for key in cses:
                res = np.copy(cl['results'][key])

                for j in range(bins):
                    bot = j*(1.0/bins)
                    top = (j + 1)*(1.0/bins)

                    binres = np.copy(res)
                    binres[maxmask < bot] = 0

                    hist[key][j] += np.sum((binres >= bot) & (binres < top))

        return hist

    def gethistrates(self, classifiers, stimuli, traces=None, appendtohist=None, likelihood=None, bins=20):
        """
        Return the false positive histogram for classifiers in classifiers. Exclude traces if random. If traces are
        included, their pupil maks will be applied to the output of the classifier.
        :param classifiers: list of classifier outputs
        :param stimuli: list of cses
        :param traces: trace2p files for each classifier, only if not random
        :param likelihood: A vector of likelihood values
        :param bins: number of histogram bins
        :param appendtohist: append the results to an existing dict of histograms (for combining across days/days)
        :return: dict of histograms for each stimulus
        """

        # Append to the previous histogram, or initialize
        hist = appendtohist
        if hist is None:
            hist = {key: np.zeros(bins) for key in stimuli}

        # Iterate over listed classifiers
        for i, cl in enumerate(classifiers):
            # Random events for a single classifier

            for key in stimuli:
                res = np.copy(cl['results'][key])
                if likelihood is not None:
                    res[cl['likelihood'][key] < likelihood] = 0

                # If need be (real results), cut by pupil diameter
                if traces is not None:
                    mask = traces[i].inactivity()
                    res = res[mask]
                hist[key] += np.histogram(res, bins, [0, 1])[0]

        return hist