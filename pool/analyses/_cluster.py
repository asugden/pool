# Moved to pool: 180817
# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import community
import networkx as nx
import numpy as np

from flow import netcom
from . import base


class Cluster(base.AnalysisBase):
    def _run_analyses(self):
        out = {}

        # Get visually driven neurons
        vdrive = self.get_driven_cells()

        # Spontaneous correlations
        sweights = self.analysis('spontcorr-allall')
        nodes, edges = self.convert_edge_weights(sweights, vdrive['all'])
        self.out['cluster-coeff-spont-all'] = netcom.nxgraph(nodes, edges).connectivity()

        for cs in ['plus', 'neutral', 'minus']:
            nodes, edges = self.convert_edge_weights(sweights, vdrive[cs])
            self.out['cluster-coeff-spont-%s' % cs] = netcom.nxgraph(nodes, edges).connectivity()

            nweights = self.analysis('noise-correlation-%s' % cs)
            nodes, edges = self.convert_edge_weights(nweights, vdrive[cs])
            self.out['cluster-coeff-%s' % cs] = netcom.nxgraph(nodes, edges).connectivity()


    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    sets = ['cluster-coeff-spont-%s' % cs for cs in ['plus', 'neutral', 'minus']] + \
           ['cluster-coeff-spont-all'] + \
           ['cluster-coeff-%s' % cs for cs in ['plus', 'neutral', 'minus']] + \
           []
    across = 'day'
    updated = '180825'

    def get_driven_cells(self, threshold=50):
        """
        Get all visually driven cells as a dict across cses and an added
        total stimulus that contains all driven cells.

        Returns
        -------
        dict
            All visually driven cells split by stimulus
        """

        vdriven = {}
        all = None
        for cs in ['plus', 'neutral', 'minus']:
            vdriven[cs] = self.analysis('visually-driven-%s' % cs) > threshold

            if all is None:
                all = np.zeros(len(vdriven[cs])) > 1
            all = np.bitwise_or(all, vdriven[cs])

        vdriven['all'] = all

        return vdriven

    @staticmethod
    def convert_edge_weights(weights, nodemask):
        """
        Convert a matrix of connection strengths into the edge weights used to define the graph

        Parameters
        ----------
        weights : matrix of ncells x ncells
            Should be noise correlations or spontaneous correlations
        nodemask : boolean array of ncells
            Shows which nodes should be included

        Returns
        -------
        nodes, edges, ncells
            A tuple of nodes, edges, and ncells to be passed to netcom's nxgraph

        """

        ncells = len(nodemask)
        nodes = np.arange(ncells)[nodemask]

        weights[np.invert(np.isfinite(weights))] = -1
        edges = []
        for i, c1 in enumerate(nodes):
            for c2 in nodes[i+1:]:
                if weights[c1, c2] != np.nan and weights[c1, c2] > 0:
                    edges.append((c1, c2, weights[c1, c2]))

        return nodes, edges