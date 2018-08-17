# Updated: 170330
# Data will be passed in a dict with the list of days for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

class HMMBehavior(object):
    def __init__(self, data):
        self.out = {}

        for run in data['train']:
            t2p = self.trace2p(run)

            for cs in ['plus', 'neutral', 'minus']:
                if 'n-stimuli-%s' % cs not in self.out:
                    self.out['n-stimuli-%s' % cs] = 0

                self.out['n-stimuli-%s'%cs] += len(t2p.csonsets(cs))
                if cs == 'plus':
                    self.out['n-stimuli-%s'%cs] += len(t2p.csonsets('pavlovian'))

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = ['n-stimuli-%s' % cs for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180207'

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

