#imports
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd




class Network():
    #-----------------------------------------------------------------------------------------------#
    '''
    Graph Generation Mechanism. Encodes player information within graph structure
    PARMS:
        num_nodes: int, scalar value determining the size of the underlying network
        prob_edges: float (0,1], Bounded probabiltiy for edge creation in random graph.
        pr_selfish: float (0,1], Bounded probability determining distribution of player strategy types.
        p_star: float [0,1]

    RETURNS:
        Graph with node attributes set

    EXAMPLE:
        #Graph Generation
        num_nodes = 5
        prob_edges = 0.5

        #Node Attributes
        pr_selfish = 0.5
        pr_altrustic = 1 - pr_selfish
        p_star = 0.20

        G, table = Network(num_nodes, prob_edges, pr_selfish, p_star).make_random_graph(plot=True, print_table=True)

        *table returns the following. Graph also outputted via matplotlib

                                  node       strat                 p_act     neighbors degree
                    0   v0     selfish  [0.1, 0.4, 0.2, 0.3]           [4]      1
                    1   v1  altruistic  [0.1, 0.2, 0.4, 0.3]        [2, 4]      2
                    2   v2  altruistic  [0.1, 0.2, 0.4, 0.3]        [1, 4]      2
                    3   v3  altruistic  [0.1, 0.2, 0.4, 0.3]           [4]      1
                    4   v4  altruistic  [0.1, 0.2, 0.4, 0.3]  [0, 1, 2, 3]      4
    '''
    #-----------------------------------------------------------------------------------------------#
    def __init__(self, num_nodes, prob_edges, pr_selfish, p_star):

        self.num_nodes = num_nodes
        self.prob_edges = prob_edges
        self.pr_selfish = pr_selfish
        self.p_star = p_star
        self.p_act = {
            'selfish':
            np.round([p_star / 2, 2 * p_star, p_star, 1 - 3.5 * p_star], 2),
            'altruistic':
            np.round([p_star / 2, p_star, 2 * p_star, 1 - 3.5 * p_star], 2)
        }

    #Set node_attributes (assign strategy and strategy vector p_act)
    def set_node_atts(self, G):
        '''
        PARMS:
            G: nx graph, A graph G(V,E)
        RETURNS:
            Graph with node attributes set
        '''
        assert self.pr_selfish <= 1. and self.pr_selfish >= 0., 'Invalid probability. pr_selfish must be [0.,1.]'

        #Randomly assign strategy profiles according to predfined probabiltiy pr_selfish
        #Append node attributes as a dictionary of randomly chosen strat types per |V|
        label = np.random.choice(list(self.p_act.keys()), self.num_nodes,
                                 self.pr_selfish)
        labels = {}
        for i, j in enumerate(label):
            labels[i] = j, self.p_act[j]

        nx.set_node_attributes(G, labels, 'atts')

        return G

    def set_edge_weights(self, G):
        '''
        PARMS:
            G: nx graph
        Returns:
            G: A weighted graph with edge weights determined via similarity between node strategy vectors

        '''
        for i in G.edges():
            a = np.round(G.nodes[i[0]]['atts'][1], 2)
            b = np.round(G.nodes[i[1]]['atts'][1], 2)
            #Compute Cos Similarity
            sim = np.round(np.dot(a, b) / (norm(a) * norm(b)),
                           2)  #Round for visualization of edge weights
            G[i[0]][i[1]]['weight'] = sim
        return G

    #Don't want disjoint commununities.
    #All edges must have a degree >= 1
    def make_random_graph(self, plot=True, print_table=True):
        '''
        PARMS:
            num_nodes: int, unbounded. |V| determines size of network
            prob_edges: float, [0,1.] determines the prob of adding edge between two nodes
            p_act: dictionary, contains strategy profile probability vectors for action space sampling
            pr_selfish: float, [0,1]: determines the distribution of strategy types within the network

        RETURNS:
            A Random Graph with node attributes set and edge weights set.

        '''
        assert self.prob_edges <= 1. and self.prob_edges >= 0., 'Invalid edge probability. prob_edges must be [0.,1.]'
        connected = False
        while not connected:
            G = nx.gnp_random_graph(self.num_nodes, self.prob_edges)
            connected = nx.is_connected(G)

        assert len(list(nx.connected_components(G))) == 1, 'Graph is Disjoint'

        G = self.set_node_atts(G)
        G = self.set_edge_weights(G)

        if print_table:
            test = {}
            for i, j in enumerate(G.nodes):
                test[i] = ('v{}'.format(i), G.nodes[i]['atts'][0],
                           G.nodes[i]['atts'][1], [j for j in G.neighbors(i)],
                           G.degree(i))

            cols = ['node', 'strat', 'p_act', 'neighbors', 'degree']
            df = pd.DataFrame.from_dict(test).T
            df.columns = cols
            print(df)

        if plot:
            color_map = []
            for i in G:
                if G.nodes[i]['atts'][0] == 'selfish':
                    color_map.append('red')
                else:
                    color_map.append('green')
            pos = nx.spring_layout(G, k=0.5, iterations=100)
            plt.figure(3, figsize=(10, 5))
            nx.draw(G, pos, node_color=color_map)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edge_labels(G, pos, font_size=7)
            plt.grid()
            plt.axis(True)
            plt.show()
        return G, test
