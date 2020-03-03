import os
from utils.graphinit import Network
from agent import agent
from env import environment
import warnings
import sys
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

warnings.filterwarnings("ignore", category=UserWarning)
app = dash.Dash(__name__)
server = app.server

if __name__ == '__main__':
    #Graph Generation
    num_nodes = int(sys.argv[1])
    prob_edges = float(sys.argv[2])
    #Node Attributes
    pr_selfish = float(sys.argv[3])
    p_star = float(sys.argv[4])
    pr_altrustic = 1 - pr_selfish
    t_max = 400

    G, table = Network(num_nodes, prob_edges, pr_selfish,
                    p_star).make_random_graph(plot=True, print_table=True)


    env = environment(G, t_max= t_max)
    env.set_env()
    env.play()
    df = env.output_logs(nodes = num_nodes, edgeprob = prob_edges, p_star = p_star, pselfish= pr_selfish, t_max= t_max)
    app.run_server(debug=True)
