# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import numpy as np

class NEvents(object):
    def __init__(self, data):
        self.out = {}
        etotal = 0
        etothungry = 0
        for cs in ['plus', 'neutral', 'minus']:
            for cmin in np.arange(0.05, 1.01, 0.05):
                self.out['nevents-%.2f-%s' % (cmin, cs)] = 0
                self.out['nevents-hungry-%.2f-%s' % (cmin, cs)] = 0

                for run in data['sated']:
                    evs = self.analysis('event-peaks-run%i-%.2f-%s' % (run, cmin, cs))
                    if evs is not None:
                        self.out['nevents-%.2f-%s'%(cmin, cs)] += len(evs)

                for run in data['hungry']:
                    evs = self.analysis('event-peaks-run%i-%.2f-%s'%(run, cmin, cs))
                    if evs is not None:
                        self.out['nevents-hungry-%.2f-%s'%(cmin, cs)] += len(evs)

            etotal += self.out['nevents-0.10-%s' % cs]
            etothungry += self.out['nevents-hungry-0.10-%s'%cs]

        for cs in ['plus', 'neutral', 'minus']:
            if etotal == 0:
                self.out['event-frac-%s'%cs] = np.nan
            else:
                self.out['event-frac-%s'%cs] = float(self.out['nevents-0.10-%s' % cs])/etotal

            if etothungry == 0:
                self.out['event-frac-hungry-%s'%cs] = np.nan
            else:
                self.out['event-frac-hungry-%s'%cs] = float(self.out['nevents-hungry-0.10-%s'%cs])/etothungry


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['classifier']
    sets = [['nevents-%.2f-%s'%(cmin, cs), 'nevents-hungry-%.2f-%s'%(cmin, cs)]
            for cs in ['plus', 'neutral', 'minus']
            for cmin in np.arange(0.05, 1.01, 0.05)] + \
           [['event-frac-%s'%cs, 'event-frac-hungry-%s'%cs]
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180207'

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    # ================================================================================== #
    # ANYTHING YOU NEED


