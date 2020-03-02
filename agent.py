from __future__ import print_function
import numpy as np
from collections import Counter
import itertools
from tqdm import tqdm
import pandas as pd

class agent(object):
    #-----------------------------------------------------------------------------------------------#
    '''
    Agent Generation Mechanism. Extracts node attribute information from graph structure

    PARMS:
        node_number: int, assigning variable name to agents. Just use index position
        G: A networkx graph structure; G(V,E)
        env: an environment. Environment and Agents will communicate and pass back and forth state information
        seed: int, used for random number generator


    RETURNS:
                                env(G(V,E), Agents(G(V,E), env))
                                agents operate within the environment class.
    '''

    #-----------------------------------------------------------------------------------------------#
    def __init__(self, node_number, G, env, seed=1, enable_print=1):
        self.node_number = node_number                  
        self.nodeData = env.nodeData  
        self.neighbors = self.nodeData[self.node_number][2]
        self.seed = np.random.seed(seed)
        self.letters = {
            'letters_initial': [],
            'letters_stolen': [],
            'letters_received': []
        }
        self.tw = []
        self.target_word = []
        self.l_union_vi = []
        self.letters_needed = []
        self.corpus = []
        self.actions_taken = 0
        self.words_formed = 0
        self.action_history = []
        self.logs = pd.DataFrame()
    #-----------------------------------------------------------------------------------------------#   
    def get_init_hand(self, env):
        #------UGet initial hand prior to game start.-----#
        self.letters['letters_initial'] = np.random.choice(env.alphabet, 5)   
        env.current_state[self.node_number] = self.letters
    #-----------------------------------------------------------------------------------------------#
    def get_target_word(self, env):
        #After c^possible is instantiated and set by the env, we use an augmented form of hamming distance
        matches = []                                         # Matches stores lengths from current to target
        len_current = len(self.l_union_vi)
        for i in self.corpus:
            matches.append(len_current - len([j for j in self.l_union_vi if j in i]))
        
        self.tw = tword = self.corpus[np.argmin(matches)]              # Find Argmin distance to target
        self.corpus.remove(tword)                            # Remove from corpus so we don't resample
        self.target_word = [i for i in tword]                # characterize word 
        self._find_needed_letters()                          # Find needed letters to new target
    #-----------------------------------------------------------------------------------------------#
    def _merge_letters(self):
        #------Returns a set of all letters, including non-unique entries-----#
        self.l_union_vi = list(itertools.chain.from_iterable(self.letters.values()))
    #-----------------------------------------------------------------------------------------------#
    def _remove_letters(self):
        #------Remove stolen or received letters that intersect with target word after forming word-----#
        stolen_intersect = np.intersect1d(self.letters['letters_stolen'], self.target_word)
        received_intersect = np.intersect1d(self.letters['letters_received'], self.target_word)
        
        for i in received_intersect:
            self.letters['letters_received'].remove(i)            # Remove Inter. from received letter set
        for i in stolen_intersect:                      
            self.letters['letters_stolen'].remove(i)              # Remove Inter. from stolen letter set 
    #-----------------------------------------------------------------------------------------------#    
    def _find_needed_letters(self):
        #------Complement of intersection of existing letter sets-----#
        self._merge_letters()                                                 # Union over all letters
        self.letters_needed = np.setdiff1d(self.target_word, self.l_union_vi) # Complement between union and target intersect
    #-----------------------------------------------------------------------------------------------#
    def _update_state_info(self, env):
        #------Update State Information, Locally-----#
        self._merge_letters()                                     # Remerge and update lunion
        self._find_needed_letters()                               # Re-find needed letters
    
        env.historical_states[env.time] = env.current_state       # Send current state to historical state
        env.current_state[self.node_number] = self.letters        # push new state to current state
        env.time += 1                                             # increment global timer
        self.actions_taken += 1                                   # increment local timer
    #-----------------------------------------------------------------------------------------------#
    def _log(self, env, initial=False):
        #store a data dict of current state information
        logs = {
            'node': self.node_number,                            # node number / index
            'strategy': self.nodeData[self.node_number][0],      # strategy type
            'action_vector': self.nodeData[self.node_number][1], # p_act according to strategy type
            'neighbors': self.nodeData[self.node_number][2],     # neighbors of agent v_i
            'degree': self.nodeData[self.node_number][3],        # degree (out) of agent v_i  
            'target' : self.tw,
            'avail_letters' : self.l_union_vi,                   # avail letters. Union over letter sets
            'letters_needed' : self.letters_needed,              # letters needed at t. 
            'global_time': env.time - 1,                         # global time counter (number of total agent moves)
            'action_number': self.actions_taken,                 # local time counter (actions taken for agent v_i)
            'action': self.action_history[-1][0],                # last action taken
            'action_success': self.action_history[-1][1],        # last action success/failure, boolean
            'words_formed': self.words_formed                    # running counter of words formed
        }
        #append current state information to a running log of local state information. 
        df = pd.DataFrame.from_dict(logs, orient='index').T 
        self.logs = self.logs.append(df)
    #-----------------------------------------------------------------------------------------------#
    def _agent_form_word(self, env):
        if len(self.letters_needed) == 0:
            env.words_formed.append(("".join(self.target_word), self.node_number))  # global WFormed Counter
            self.words_formed += 1                                                  # local WFormed Coutner
            self._remove_letters()                          # Remove Letters used to form target word           
            self.get_target_word(env)                       # Get new target word
            self.action_history.append(('form', True))
        else:
            self.action_history.append(('form', False))
    #-----------------------------------------------------------------------------------------------#
    def _agent_steal_letter(self, env):
        if len(self.letters_needed) != 0:                                   # if letters need is empty, word is formable
            choose_letter_to_steal = np.random.choice(self.letters_needed)  # else, select random letter needed 
        else:
            choose_letter_to_steal = None                                   


        if choose_letter_to_steal != None:
            for k, v in env.agents.items():
                if k in self.neighbors:
                    if choose_letter_to_steal in v.l_union_vi:              
                        self.letters['letters_stolen'].append(choose_letter_to_steal)  
                        break                                                          # Only want to steal a single letter
            self.action_history.append(('steal', True))
        else:
            self.action_history.append(('steal', False))
    #-----------------------------------------------------------------------------------------------#
    def _agent_pass_letter(self, env):
        self.action_history.append('pass letter')

        choose_letter_to_pass = np.random.choice(self.letters['letters_initial'])
        if choose_letter_to_pass != None:
            for k, v in env.agents.items():
                if k in self.neighbors:
                    if choose_letter_to_pass not in v.l_union_vi:
                        env.agents[k].letters['letters_received'].append(
                            choose_letter_to_pass)  #update local state
                        break
            self.action_history.append(('pass', True))
        else:
            self.action_history.append(('pass', False))
    #-----------------------------------------------------------------------------------------------#
    def _agent_think(self, env):
        self.action_history.append(('think', True))
    #-----------------------------------------------------------------------------------------------#
    def take_action(self, env):
        n_Data = self.nodeData[self.node_number]
        p_act = n_Data[1]
        action_sampling = np.random.choice(env.action_space, p=p_act)

        print('-' * 50)
        print('action to node: {}'.format(self.node_number))
        print('Move Number: {} , Env Clock: {}'.format(self.actions_taken,  env.time))


        if action_sampling == env.action_space[0]:  #form_word
            self._agent_form_word(env)
        elif action_sampling == env.action_space[1]:  #steal_letter
            self._agent_steal_letter(env)
        elif action_sampling == env.action_space[2]:  #pass_letter
            self._agent_pass_letter(env)
        elif action_sampling == env.action_space[3]:  #think / null action
            self._agent_think(env)
            
        self._update_state_info(env)
        self._log(env)
        print('-' * 50)
