import os
from utils.graphinit import Network
from agent import agent
from env import environment


if __name__ == '__main__':
    #Graph Generation
    num_nodes = 5
    prob_edges = 0.5

    #Node Attributes
    pr_selfish = 0.5
    pr_altrustic = 1 - pr_selfish
    p_star = 0.20


    G, table = Network(num_nodes, prob_edges, pr_selfish,
                      p_star).make_random_graph(plot=True, print_table=True)
    env = environment(G, t_max= 400)
    env.set_env()
