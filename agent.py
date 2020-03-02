from __future__ import print_function
import numpy as np
from collections import Counter
import itertools
from tqdm import tqdm


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
    def __init__(self, node_number, G, env, seed=1, enable_print=1):
        self.node_number = node_number
        self.nodeData = env.nodeData                                 #key (node number): (strategy, p_act, neighbors, degree)
        self.seed = np.random.seed(seed)
        self.letters = {
            'letters_initial': [],
            'letters_stolen': [],
            'letters_received': []
        }
        self.target_word = []
        self.l_union_vi = []
        self.letters_needed = []
        self.corpus = []
        self.actions_taken = 0
        self.words_formed = 0

    def _merge_letters(self):
        #-----------------------------------------------------------------------------------------------#
        '''
        Merge all letters in agent.letters
        This returns a set of all letters, including non-unique entries
        '''
        #-----------------------------------------------------------------------------------------------#
        self.l_union_vi = list(itertools.chain.from_iterable(self.letters.values()))

    def _find_needed_letters(self):
        #-----------------------------------------------------------------------------------------------#
        '''
        Find the letters needed to complete a target word. Calc difference between two independent sets.
        '''
        #-----------------------------------------------------------------------------------------------#
        self._merge_letters()
        self.letters_needed = np.setdiff1d(self.target_word, self.l_union_vi)

    def _update_state_info(self,env):
        self._merge_letters() #Remerge and update lunion
        self._find_needed_letters() #re-find needed letters
        env.historical_states[env.time] = env.current_state
        env.current_state[self.node_number] = self.letters
        env.time += 1
        self.actions_taken += 1
        print('state info updated')

    def get_init_hand(self, env):
        #-----------------------------------------------------------------------------------------------#
        '''
        Get initial hand prior to game start.
        t=0 at this point. We randomly choose letters over the alphabet uniformly.
        Assign back to dict in environment containing local state information.
        '''
        #-----------------------------------------------------------------------------------------------#
        self.letters['letters_initial'] = np.random.choice(
            env.alphabet, 5)  #k is the length of each element in corpus
        env.current_state[self.node_number] = self.letters

    def get_target_word(self, env):
        #-----------------------------------------------------------------------------------------------#
        '''
        After c^possible is instantiated and set by the env, we use an augmented form of hamming distance
        to determine a target word that is the closest to us.
        '''
        #-----------------------------------------------------------------------------------------------#
        matches = []
        for i in self.corpus:
            matches.append(
                len(self.l_union_vi) -
                len([j for j in self.l_union_vi if j in i]))
        tword = self.corpus[np.argmin(matches)]
        self.corpus.remove(tword)
        self.target_word = [i for i in tword]
        self._find_needed_letters()

    def _agent_form_word(self,env):
        #-----------------------------------------------------------------------------------------------#
        '''
        Need to update to remove letters from Lrecieved and Lstolen if used to form a word
        '''
        #-----------------------------------------------------------------------------------------------#
        print('-' * 50)
        assert env.time < env.time_max, 'Maximum time allowed has been reached'
        print('action to node: {}'.format(self.node_number))
        print('Move Number: {} , Env Clock: {}'.format(self.actions_taken, env.time))

        print('taking action: Form Word')
        if len(self.letters_needed) == 0:
            env.words_formed.append(("".join(self.target_word),self.node_number)) #global counter
            self.get_target_word(env) #get new target word
        else:
            print('unable to form word. Need additional letters.')
        self._update_state_info(env)
        print('-' * 50)



    def _agent_steal_letter(self,env):
        #-----------------------------------------------------------------------------------------------#
        '''
        Find letters needed to complete the target word.
        Randomly selects a single letter.
        Finds neighbors of current node and scans their union set to see if this letter exists. If it does, steals letter.
        Update env.current state and pop current state to historical state. Increment timer
        Need to update so a letter stolen from non init dist is depleted
        '''
        #-----------------------------------------------------------------------------------------------#
        print('-' * 50)
        assert env.time < env.time_max, 'Maximum time allowed has been reached'
        print('action to node: {}'.format(self.node_number))
        print('Move Number: {} , Env Clock: {}'.format(self.actions_taken, env.time))

        print('taking action: steal letter')
        neighbors = self.nodeData[self.node_number][2]
        print('available neighbors: {}'.format(neighbors))
        letters_needed = self.letters_needed
        print(f'letters needed to reach the target {letters_needed}')
        if len(letters_needed) != 0:
            choose_letter_to_steal = np.random.choice(letters_needed)
        else: choose_letter_to_steal = None

        if choose_letter_to_steal != None:
            print(f'randomly selecting letter to steal: {choose_letter_to_steal}')
            for k, v in env.agents.items():
                if k in neighbors:
                    if choose_letter_to_steal in v.l_union_vi:
                        print(
                            f'neighbor {env.agents[k].node_number} : neighbor hand {env.agents[k].l_union_vi}'
                        )
                        self.letters['letters_stolen'].append(
                            choose_letter_to_steal)  #update local state
                        print('letter successfully stolen from node {}'.format(k))
                        break
        else:
            print('unable to steal the chosen letter from k-hop neighbors')

        self._update_state_info(env)
        print('-' * 50)

    def _agent_pass_letter(self,env):
        #-----------------------------------------------------------------------------------------------#
        '''
        Need to update so a letter passed from non init dist is depleted
        '''
        #-----------------------------------------------------------------------------------------------#
        print('-' * 50)
        assert env.time < env.time_max, 'Maximum time allowed has been reached'
        print('action to node: {}'.format(self.node_number))
        print('Move Number: {} , Env Clock: {}'.format(self.actions_taken, env.time))
        print('taking action: pass letter')
        neighbors = self.nodeData[self.node_number][2]
        print('available neighbors: {}'.format(neighbors))
        choose_letter_to_pass = np.random.choice(self.letters['letters_initial'])
        if choose_letter_to_pass != None:
            print(f'randomly selecting letter to pass: {choose_letter_to_pass}')
            for k, v in env.agents.items():
                if k in neighbors:
                    if choose_letter_to_pass not in v.l_union_vi:
                        print(
                            f'neighbor {env.agents[k].node_number} : neighbor hand {env.agents[k].l_union_vi}'
                        )
                        env.agents[k].letters['letters_received'].append(
                            choose_letter_to_pass)  #update local state
                        print('letter successfully passed from node {} to node {}'.format(self.node_number, k))
                        break
        else:
            print('unable to steal the chosen letter from k-hop neighbors')

        self._update_state_info(env)
        print('-' * 50)

    def _agent_think(self,env):
        #-----------------------------------------------------------------------------------------------#
        '''
        '''
        #-----------------------------------------------------------------------------------------------#
        print('-' * 50)
        assert env.time < env.time_max, 'Maximum time allowed has been reached'
        print('action to node: {}'.format(self.node_number))
        print('Move Number: {} , Env Clock: {}'.format(self.actions_taken, env.time))

        print('taking action: think')
        self._update_state_info(env)
        print('-' * 50)

    def take_action(self,env):
        #-----------------------------------------------------------------------------------------------#
        '''
        '''
        #-----------------------------------------------------------------------------------------------#
        n_Data = self.nodeData[self.node_number]
        # node = self.node_number
        # strategy = n_Data[0]
        p_act = n_Data[1]
        # neighbors = n_Data[2]
        action_sampling = np.random.choice(env.action_space, p=p_act)

        if action_sampling == env.action_space[0]:  #form_word
            self._agent_form_word(env)
        if action_sampling == env.action_space[1]:  #steal_letter
            self._agent_steal_letter(env)
        if action_sampling == env.action_space[2]:  #pass_letter
            self._agent_pass_letter(env)
        if action_sampling == env.action_space[3]:  #think / null action
            self._agent_think(env)
