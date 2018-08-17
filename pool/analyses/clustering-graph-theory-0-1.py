# Data will be passed in a dict with the list of runs for training, spontaneous, running, and for across='run', run
# Two methods are automatically injected: trace2p and classifier.
# It is required to return via the function get

import community
import networkx as nx
import numpy as np


class Clustering(object):
    def __init__(self, data):
        self.out = {}

        self.vdriven = {}
        for cs in ['plus', 'neutral', 'minus']:
            self.vdriven[cs] = self.analysis('visually-driven-%s' % cs) > 50

        for cs in ['plus', 'neutral', 'minus']:
            nodes, edges, ncells = self.nodeedges(cs, '-0-1')
            self.out['graph-clustering-0-1-%s' % cs] = self.nodeclusters(nodes, edges, ncells)

            nodes, edges, ncells = self.nodeedges(cs, '-nolick')
            self.out['graph-clustering-nolick-%s' % cs] = self.nodeclusters(nodes, edges, ncells)

            nodes, edges, ncells = self.nodeedges(cs, '-0-1.5')
            self.out['graph-clustering-0-1.5-%s' % cs] = self.nodeclusters(nodes, edges, ncells)

            nodes, edges, ncells = self.nodeedges(cs, '-0-1.5-decon')
            self.out['graph-clustering-0-1.5-decon-%s' % cs] = self.nodeclusters(nodes, edges, ncells)

    # ================================================================================== #
    # REQUIRED PARAMETERS AND INJECTED FUNCTIONS

    # Set the screening parameters by which protocols are selected
    # Screening by protocol is required. Any other screening is optional
    # Options for requires include:
    #	classifier: whether a classifier will be required or not

    requires = ['']
    sets = [['graph-clustering-0-1-%s'%cs, 'graph-clustering-nolick-%s'%cs,
             'graph-clustering-0-1.5-%s'%cs, 'graph-clustering-0-1.5-decon-%s'%cs]
            for cs in ['plus', 'neutral', 'minus']]
    across = 'day'
    updated = '180526'

    def get(self):
        """
        Required function
        :return: must return dict of outputs
        """
        return self.out

    def nodeedges(self, cs='', nctype=''):
        """
        Get node labels and edge weights for days as noise correlations or spontanoues correlations (or combinations
        thereof)
        :param andb: analysis database instance
        :param mouse: mouse name, str
        :param date: date, str
        :param cs: stimulus name for noise correlations, blank for stimulus correlations
        :param combine: if True, combine stimulus and noise correlations
        :return:
        """

        if len(cs) > 0 and cs != 'spontaneous':
            corr = self.analysis('noise-correlation%s-%s' % (nctype, cs))
            vdrive = self.vdriven[cs]
            nodes = np.arange(len(vdrive))[vdrive]
        else:
            corr = self.analysis('spontaneous-correlation')
            corr[corr > 0.0038 + 3*0.0188] = np.nan
            nodes = np.arange(np.shape(corr)[0])

        corr[np.invert(np.isfinite(corr))] = -1
        corr[corr < 0] = -1
        edges = []
        ncells = np.shape(corr)[0]

        for i, c1 in enumerate(nodes):
            for c2 in nodes[i+1:]:
                if corr[c1, c2] != np.nan and corr[c1, c2] > 0:
                    edges.append((c1, c2, corr[c1, c2]))

        return nodes, edges, ncells

    def nodeclusters(self, nodes, edges, ncells):
        """
        Graph a community
        :param path: path to save graph, if blank then plt locally
        :param nodes: list of node names (list of ints)
        :param nodeclr: dict of node colors
        :param edges: list of 3-tuples of (node1, node2, weight)
        :return: None
        """

        # Add nodes and edges
        gx = nx.Graph()

        for c1 in nodes:
            gx.add_node(c1)

        for c1, c2, weight in edges:
            gx.add_edge(c1, c2, weight=weight)

        cluster = nx.clustering(gx, None, weight='weight')

        out = np.full(ncells, np.nan)
        for c1 in nodes:
            if cluster[c1] > 0:
                out[c1] = cluster[c1]

        return out

    def localnodeclusters(self, nodes, edges, ncells, within):
        """
        Graph a community
        :param path: path to save graph, if blank then plt locally
        :param nodes: list of node names (list of ints)
        :param nodeclr: dict of node colors
        :param edges: list of 3-tuples of (node1, node2, weight)
        :return: None
        """

        # Add nodes and edges
        gx = nx.Graph()

        for c1 in nodes:
            gx.add_node(c1)

        for c1, c2, weight in edges:
            gx.add_edge(c1, c2, weight=weight)

        parts = community.best_partition(gx)
        ncommunities = len(np.unique([parts.get(node) for node in gx.nodes()]))

        for c1, c2, w in gx.edges(data=True):
            com1 = parts.get(c1)
            com2 = parts.get(c2)

            # 0 out connection values
            gx[c1][c2]['between'] = 0
            for com in range(ncommunities):
                gx[c1][c2]['within-%i' % com] = 0

            if com1 == com2:
                gx[c1][c2]['within-%i' % com1] = w['weight']
            else:
                gx[c1][c2]['between'] = w['weight']

        out = np.full(ncells, np.nan)

        if len(edges) > 1:
            if within:
                cluster = [self.safecluster(gx, 'within-%i' % com) for com in range(ncommunities)]

                for c1 in nodes:
                    com1 = parts.get(c1)
                    out[c1] = cluster[com1][c1]
            else:
                cluster = self.safecluster(gx, 'between')

                for c1 in nodes:
                    out[c1] = cluster[c1]

        return out

    def safecluster(self, gx, weight):
        """
        Cluster, accounting for 0 weights
        :param gx:
        :param weight:
        :return:
        """

        total = 0
        for c1, c2, w in gx.edges(data=True):
            total += w[weight]

        if total > 0:
            return nx.clustering(gx, None, weight=weight)
        else:
            return {c1:0 for c1 in gx.nodes()}

    def clustern(self, ctype='hofer'):
        """
        Get the cluster number for each neuron
        :param ctype: cluster type, hofer or spontaneous
        :return: np array of cluster number
        """

        nodes, edges, ncells = self.nodeedges(ctype)

        gx = nx.Graph()

        for c1 in nodes:
            gx.add_node(c1)

        for c1, c2, weight in edges:
            gx.add_edge(c1, c2, weight=weight)

        parts = community.best_partition(gx)

        out = np.full(ncells, np.nan)
        for node in nodes:
            out[node] = parts.get(node)

        return out

    def getgroups(self, cs, mincount=1):
        """
        Get number of clusters for each cs
        :return: None
        """

        vdrive = self.vdriven[cs]
        clusters = self.out['cluster-n'][vdrive]
        uniquens, counts = np.unique(clusters, return_counts=True)
        return len(uniquens[counts > mincount])

def divzero(a, b):
    """
    Return np.nan if dividing by zero
    :param a:
    :param b:
    :return:
    """

    if b == 0:
        return np.nan
    else:
        return float(a)/b