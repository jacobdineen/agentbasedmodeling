import numpy as np
from collections import Counter
import itertools


class agent(object):
    #-----------------------------------------------------------------------------------------------#
    '''
    Agent Generation Mechanism. Extracts node attribute information from graph structure

    PARMS:
        node_number: int, assigning variable name to agents. Just use index position
        G: A networkx graph structure; G(V,E)
        env: an environment. Environment and Agents will communicate and pass back and forth state information
        seed: int, used for random number generator

                                env(G(V,E), Agents(G(V,E), env))
                                agents operate within the environment class.

    RETURNS:
        Graph with node attributes set
    '''
    #-----------------------------------------------------------------------------------------------#
    def __init__(self, node_number, G, env, seed=1):
        self.corpus = np.loadtxt(fname='txt/five_letter_words.txt',
                                 dtype='str')
        self.node_number = node_number
        self.nodeData = env.nodeData  #key (node number): (strategy, p_act, neighbors, degree)
        self.seed = np.random.seed(seed)
        self.letters = {
            'letters_initial': [],
            'letters_stolen': [],
            'letters_received': []
        }
        self.target_word = []

    def get_init_hand(self):
        #-----------------------------------------------------------------------------------------------#
        '''
        '''
        #-----------------------------------------------------------------------------------------------#
        self.letters['letters_initial'] = np.random.choice(
            env.alphabet, 5)  #k is the length of each element in corpus
        env.current_state[self.node_number] = self.letters

    def get_target_word(self, env):
        #-----------------------------------------------------------------------------------------------#
        '''
        '''
        #-----------------------------------------------------------------------------------------------#
        matches = []
        for i in env.corpus_possible:
            matches.append(
                len(self.letters['letters_initial']) -
                len([j for j in self.letters['letters_initial'] if j in i]))
        self.target_word = env.corpus_possible[np.argmin(matches)]

    def _agent_form_word(self):
        #-----------------------------------------------------------------------------------------------#
        '''


        '''
        #-----------------------------------------------------------------------------------------------#
        pass

    def _agent_steal_letter(self):
        #-----------------------------------------------------------------------------------------------#
        '''


        '''
        #-----------------------------------------------------------------------------------------------#
        pass

    def _agent_pass_letter(self):
        #-----------------------------------------------------------------------------------------------#
        '''

        '''
        #-----------------------------------------------------------------------------------------------#
        pass

    def _agent_think(self):
        #-----------------------------------------------------------------------------------------------#
        '''


        '''
        #-----------------------------------------------------------------------------------------------#
        pass

    def take_action(self):
        #-----------------------------------------------------------------------------------------------#
        '''
        '''
        #-----------------------------------------------------------------------------------------------#
        for k, v in self.agents.items():
            node = k
            strategy = v.nodeData[k][0]
            p_act = v.nodeData[k][1]
            neighbors = v.nodeData[k][2]
            action_sampling = np.random.choice(self.action_space, p=p_act)

            if action_sampling == self.action_space[0]:  #form_word
                pass
            if action_sampling == self.action_space[1]:  #steal_letter
                pass
            if action_sampling == self.action_space[2]:  #pass_letter
                pass
            if action_sampling == self.action_space[3]:  #think / null action
                pass
