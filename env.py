from utils.helpers import possible_words
import numpy as np
from agent import agent
import itertools
import pandas as pd

class environment(object):
    #-----------------------------------------------------------------------------------------------#
    '''
    Agent Generation Mechanism. Extracts node attribute information from graph structure

    PARMS:
        G: A networkx graph structure; G(V,E)

    '''

    #-----------------------------------------------------------------------------------------------#
    def __init__(self, G, t_max):

        self.G = G                                                   # Graph Structure, Networkx         
        self.nodes = G.nodes                                         # Graph Nodes
        self.edges = G.edges                                         # Graph Edges
        self.agents = []                                             # List of Agent Objects
        self.action_space = ['form_word', 'steal_letter',            # Action Space
                             'pass_letter', 'think']          
        self.l_init_union = []                                       # Union over all letter sets for all agents
        self.corpus = np.loadtxt(fname='txt/five_letter_words.txt',
                                 dtype='str')                        # Corpus of 5 letter words
        self.corpus_possible = []
        self.alphabet = np.loadtxt(fname='txt/alphabet_english.txt', # List of characterse in english alphabet
                                   dtype='str') 
        self.nodeData = {}                                           # Node Data passed in from Graph 
        self.current_state = {}                                      # Global state at current time step
        self.historical_states = {}  #pop from current state at t+1  # Global states (logs)
        self.words_formed = []  #by player                           # List of all words formed up to current timestep
        self.time = 0                                                # Global Timer
        self.time_max = t_max                                        # Global Time constraint
        self.logs = 0                                                # Global log file. Appended to after t = tmax

    #-----------------------------------------------------------------------------------------------#
    def _getnodeData(self):
        #------Get Node Attributes------#
        for i, j in enumerate(self.G.nodes):
            self.nodeData[i] = (self.G.nodes[i]['atts'][0],      # Strategy Type
                                self.G.nodes[i]['atts'][1],      # Probability_Act, sampling densities
                                [j for j in self.G.neighbors(i)],# List of neighbors, N(v_i)
                                self.G.degree(i))                # d_out(v_i)

    #-----------------------------------------------------------------------------------------------#
    def _find_cand_words(self):

        l_init_union = [[i for l in d.values() for i in l]
                        for d in self.current_state.values()
                        ]  #Compute Union over all init letter dist
        self.l_init_union = list(
            itertools.chain(*l_init_union))  #flatten to 1d array

        self.corpus_possible = possible_words(
            self.corpus, self.l_init_union)  #find C^possible

        for i in self.agents:
            self.agents[i].corpus = self.corpus_possible

        print(
            f'total corpus size {len(self.corpus)}, total count of possible words {len(self.corpus_possible)}'
        )

    #-----------------------------------------------------------------------------------------------#
    def _push_to_historical(self):

        self.historical_states[self.time] = self.current_state

    #-----------------------------------------------------------------------------------------------#
    def _set_init_targets(self):

        [
            self.agents[i].get_target_word(self)
            for i in range(len(self.G.nodes))
        ]
        self._push_to_historical()  #push to historical states dict
        self.time += 1  #increment time counter

    #-----------------------------------------------------------------------------------------------#
    def set_agents(self):
        #-------------------------------------------------
        #Create Agents
        #-------------------------------------------------
        agents = {}
        num_nodes = len(self.G.nodes)
        agent_names = [str(i) for i in range(len(self.G.nodes))]
        for i, j in enumerate(agent_names):
            agents[i] = agent(node_number=i, G=self.G, env=self)

        self.agents = agents

        [j.get_init_hand(self) for i, j in self.agents.items()
         ]  #-> L_init(agent) & env.current_state
        self._find_cand_words(
        )  #Find Candidate Words - C^Possible, given init letter distr
        self._set_init_targets()  #Set initial target word

    def set_env(self):
        #-----------------------------------------------------------------------------------------------#
        assert len(self.alphabet) == 26, 'Alphabet size mismatch'
        self._getnodeData()
        self.set_agents()
        print('Environment set - > Graph Created. Node attributes assigned.')

    #-----------------------------------------------------------------------------------------------#
    def output_logs(self, nodes, edgeprob, p_star, pselfish):
        self.logs = pd.concat(
            self.agents[i].logs
            for i in range(len(self.nodes))).sort_values('global_time')
        self.logs.to_csv('data/Nodes_{}_P_{}_Pstar_{}_PSelfish_{}.csv'.format(nodes, edgeprob, p_star, pselfish))

    #-----------------------------------------------------------------------------------------------#
    def play(self):
        while self.time <= self.time_max:
            for j in range(0, len(self.agents)):
                self.agents[j].take_action(self)