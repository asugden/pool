# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

from itertools import chain
import math
import numpy as np

class MutualInformation(object):
    def __init__(self, data):
        self.out = {}
        self.analyze(data)


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #   classifier: whether a classifier will be required or not

    requires = ['classifier']
    # sets = ['mutual-information', 'mutual-information-pair', 'entropy', 'entropy-pair'] + \
    #       ['marginal-%s' % (cs) for cs in ['plus', 'neutral', 'minus', 'other', 'other-running']] + \
    #       ['mutual-information-%s' % (cs) for cs in ['plus', 'neutral', 'minus']] + \
    #       ['mutual-information-pair-%s' % (cs) for cs in ['plus', 'neutral', 'minus']]
    sets = [['mutual-information-%s'%cs, 'marginal-%s'%cs]
                for cs in ['plus', 'neutral', 'minus', 'other', 'other-running', 'ensure', 'quinine']]
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

    def joint_probability(self, marginal, conditional):
        """
        Return joint probabilities from the marginal and conditional.
        """

        classes = [cl for cl in marginal]
        ncells = np.shape(marginal[classes[0]])[0]

        joint = {}
        for cs in marginal:
            joint[cs] = np.zeros((ncells, ncells, 4))
            for c in range(ncells):
                joint[cs][c, :, 0] = conditional[cs][c, :, 0]*marginal[cs][c, 0]
                joint[cs][c, :, 1] = conditional[cs][c, :, 1]*marginal[cs][c, 0]
                joint[cs][c, :, 2] = conditional[cs][c, :, 2]*marginal[cs][c, 1]
                joint[cs][c, :, 3] = conditional[cs][c, :, 3]*marginal[cs][c, 1]

        return joint

    def individual_cs(self, gm, cs):
        """
        Get the mutual information specific to the marginal, i.e. in the style of Naive-Bayes,
        for a single stimulus
        :param run: 
        :param cs: 
        :return: 
        """

        csprior = 0.5
        noncsprior = 1.0 - csprior;

        marg = gm['marginal']
        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        noncs = [cl for cl in classes if cl != cs]

        out = np.zeros(ncells)
        if cs not in marg: return out

        for c in range(ncells):
            # Sum over states
            for tf in range(2):
                # P(tf|n union m union o union o-running)
                ptfnoncs = 0  # Probability of state tf for non-cs stimuli
                for cl in noncs: ptfnoncs += marg[cl][c, tf]*(1.0/len(noncs))*noncsprior
                ptfnoncs = ptfnoncs/noncsprior
                ptfcs = marg[cs][c, tf]
                ptfsum = ptfnoncs*noncsprior + ptfcs*csprior
                out[c] += ptfcs*csprior*math.log(ptfcs/ptfsum, 2) + ptfnoncs*noncsprior*math.log(ptfnoncs/ptfsum, 2)

        return out

    def individual(self, gm):
        """
        Return the mutual information for a cell, calculated across all stimuli
        :param gm: classifier 
        :return: 
        """

        marg = gm['marginal']
        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        pcl = 1.0/len(classes)  # Probability of a class

        out = np.zeros(ncells)
        for c in range(ncells):
            for cl in classes:
                for tf in range(2):
                    denom = 0
                    for clprime in classes: denom += marg[clprime][c, tf]*pcl
                    out[c] += marg[cl][c, tf]*pcl*math.log((marg[cl][c, tf])/denom, 2)

        order = [i[0] for i in sorted(enumerate(out), key=lambda x: x[1])]
        return out

    def pair(self, gm):
        """
        Return the paired mutual information
        :param gm: classifier
        :return: 
        """

        marg = gm['marginal']
        cond = gm['conditional']
        if len(cond) == 0: return []
        joint = self.joint_probability(marg, cond)

        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]

        out = np.zeros((ncells, ncells))
        pcl = 1.0/len(classes)  # Probability of a class
        for c1 in range(ncells):
            for c2 in range(ncells):
                for cl in classes:
                    for ttff in range(4):
                        denom = 0
                        for clprime in classes: denom += joint[clprime][c1, c2, ttff]*pcl
                        out[c1, c2] += joint[cl][c1, c2, ttff]*pcl*math.log((joint[cl][c1, c2, ttff])/denom, 2)

        return out

    def pair_cs(self, gm, cs):
        """
        Get the pair mutual information for a particular cs
        :param gm: 
        :param cs: 
        :return: 
        """

        csprior = 0.5
        noncsprior = 1.0 - csprior;

        marg = gm['marginal']
        cond = gm['conditional']
        if len(cond) == 0: return []
        joint = self.joint_probability(marg, cond)

        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        noncs = [cl for cl in classes if cl != cs]

        out = np.zeros((ncells, ncells))
        for c1 in range(ncells):
            for c2 in range(ncells):
                if c1 != c2:
                    for ttff in range(4):
                        pttffnoncs = 0  # Probability of state ttff for non-cs stimuli
                        for cl in noncs: pttffnoncs += joint[cl][c1, c2, ttff]*(1.0/len(noncs))*noncsprior
                        pttffnoncs = pttffnoncs/noncsprior
                        pttffcs = joint[cs][c1, c2, ttff]
                        pttffsum = pttffnoncs*noncsprior + pttffcs*csprior

                        out[c1, c2] += (pttffcs*csprior*math.log(pttffcs/pttffsum, 2) +
                                        pttffnoncs*noncsprior*math.log(pttffnoncs/pttffsum, 2))

        return out

    def individual_entropy(self, gm):
        marg = gm['marginal']
        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        pcl = 1.0/len(classes)  # Probability of a class

        class_entropy = 0
        for i in range(len(classes)):
            class_entropy += pcl*math.log(pcl, 2)
        class_entropy *= -1

        state_entropy = np.zeros(ncells)
        for c in range(ncells):
            for tf in range(2):
                clsum = 0
                for cl in classes:
                    clsum += marg[cl][c, tf]*pcl
                state_entropy[c] += clsum*math.log(clsum, 2)
        state_entropy *= -1

        return state_entropy

    def pair_entropy(self, gm):
        marg = gm['marginal']
        cond = gm['conditional']
        if len(cond) == 0: return []
        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        pcl = 1.0/len(classes)  # Probability of a class

        class_entropy = 0
        for i in range(len(classes)):
            class_entropy += pcl*math.log(pcl, 2)
        class_entropy *= -1

        state_entropy = np.zeros((ncells, ncells))
        for c1 in range(ncells):
            for c2 in range(ncells):
                for ttff in range(4):
                    clsum = 0
                    for cl in classes:
                        clsum += cond[cl][c1, c2, ttff]*pcl
                    state_entropy[c1, c2] += clsum*math.log(clsum, 2)
        state_entropy *= -1

        return state_entropy

    def analyze(self, data):
        run = data['sated'][0]
        classifier = self.classifier(run)
        # self.out['mutual-information'] = self.individual(classifier)
        # self.out['mutual-information-pair'] = self.pair(classifier)
        # self.out['entropy'] = self.individual_entropy(classifier)
        # self.out['entropy-pair'] = self.pair_entropy(classifier)

        # for cs in ['plus', 'neutral', 'minus']:
        #   self.out['mutual-information-pair-%s'%(cs)] = self.individual_cs(classifier, cs)

        for cs in ['plus', 'neutral', 'minus', 'other', 'other-running', 'ensure', 'quinine']:
            if cs not in classifier['marginal']:
                self.out['marginal-%s'%cs] = None
                self.out['mutual-information-%s'%cs] = None
            else:
                self.out['marginal-%s'%cs] = classifier['marginal'][cs][:, 0] if cs in classifier['marginal'] else []
                self.out['mutual-information-%s'%cs] = self.individual_cs(classifier, cs)
