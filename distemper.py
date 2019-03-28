"""
Distemper simulation for RL Agent.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
import json

import networkx as nx

from simulation import Simulation

from sklearn.preprocessing import OneHotEncoder

FLATTEN = lambda l: [item for sublist in l for item in sublist]

class Distemper(gym.Env):
    """
    Description:
        A simulation of the canine distemper virus.

    Observation:
        Type: Box(4)
        Num Observation                 Min         Max
        0   Number of Infected          0           Inf
        1   Kenne States                ?           ?
        2   Number of Intakes           0           Inf

    Actions:
        Type: Discrete(2)
        Num Action
        0   Move to next iteration
        1   Increment i
        2   Increment j
        3   Swap i and j contents

    Reward:
        10-(Number of Infected)

    Starting State:
        Empty Kennel Layout

    Episode Termination:
        31 Days (=744 hours(i.e. simulation steps))
        Consider early stopping if infection ratio is too high (Greater than 0.5)
    """

    def __init__(self,**kwargs):

        self._reset_params()
        self.simulation = Simulation(self.params,
                                     spatial_visualization=False,
                                     aggregate_visualization=False,
                                     return_on_equillibrium=True)
        self.state_encoder = OneHotEncoder(handle_unknown='error', sparse=False)
        self.state_encoder.fit([[x, 1] for x in self.simulation.disease.id_map.values()])
        self.num_nodes = len(self.simulation.disease.graph.nodes)
        self.num_states = len(self.simulation.disease.id_map.values())
        self.action_space = spaces.Discrete(8)
        self.num_states = len(self.state_encoder.transform([[0, 1]])[0])
        self.observation_space = spaces.Discrete(self.num_states*self.num_nodes+4)

        #self.reward_bias = 5.0 if kwargs.get('reward_bias') is None else kwargs.get('reward_bias')
        
        # Incentive method 1
        self.bonus_reward = 0. if kwargs.get('bonus_reward') is None else kwargs.get('bonus_reward')
        
        # Incentive method 2
        # turn_around_rate: force a simulation update for every turn_around_rate of non-0th actions taken
        self.turn_around_rate = 400 if kwargs.get('turn_around_rate') is None else kwargs.get('turn_around_rate') 
        self.turn_around_counter = 0
        
        self.incentive_methods = [0, 1]
        
        # Action stats
        self.actions_history = []
        self.turn_around_actions_history = []

        self.reward_bias = 100.0
        
        # Randomly initialize i and j
        self.start_i = 0#np.random.randint(0, self.num_nodes)
        self.start_j = 0#np.random.randint(0, self.num_nodes)
        # If they happen to be equal, adjust j randomly up or down 1 (circling around to 0 if needed)
        if self.start_i == self.start_j:
            adjustment = -1 if bool(np.random.randint(2)) else 1
            self.start_j = (self.start_j + adjustment) % self.num_nodes
        self.i = self.start_i
        self.j = self.start_j

        # Rotation
        self.k = 0
        self.r = 0

        self.swaps_this_hour = 0
        self.max_expected_swaps = 10

        self.components = [list(x) for x in nx.connected_components(self.simulation.disease.graph)]
        
        self.seed()

        self.state, _ = self._get_state_from_simulation()

    def _reset_params(self):
        
        if os.path.exists('./sim_params.json'):
            with open('./sim_params.json') as f:
                self.params = json.load(f)
                print("Loaded ./sim_params.json"+"-"*30)
        else:
            # Note: all probabilities are in units p(event) per hour
            self.params = {
                # Intake Probabilities (Note, 1-sum(these) is probability of no intake)
                'pSusceptibleIntake': 0.125,
                'pInfectIntake': 0.02,
                'pSymptomaticIntake': 0.01,
                'pInsusceptibleIntake': 0.05,

                # Survival of Illness
                'pSurviveInfected': 0.025,
                'pSurviveSymptomatic': 0.025,

                # Alternate Death Rate
                'pDieAlternate': 0.001,

                # Discharge and Cleaning
                'pDischarge': 0.05,
                'pCleaning': 0.9,

                # Disease Refractory Period
                'refractoryPeriod': 3.0*24.0,

                # Death and Symptoms of Illness
                'pSymptomatic': 0.04,
                'pDie': 0.05,

                # Infection Logic
                'infection_kernel': [0.05, 0.01],
                'infection_kernel_function': 'lambda node, k: k*(1-node[\'occupant\'][\'immunity\'])',

                # Immunity Growth (a0*immunity+a1)
                # (1.03, 0.001 represents full immunity in 5 days)
                #'immunity_growth_factors': [1.03, 0.001],
                'immunity_growth_factors': [0.0114, 0.0129, 0.0146, 0.0166, 0.0187, 0.0212, 0.0240,
                                            0.0271, 0.0306, 0.0346, 0.0390, 0.0440, 0.0496, 0.0559,
                                            0.0629, 0.0707, 0.0794, 0.0891, 0.0998, 0.1117, 0.1248,
                                            0.1392, 0.1549, 0.1721, 0.1908, 0.2109, 0.2326, 0.2558,
                                            0.2804, 0.3065, 0.3338, 0.3623, 0.3918, 0.4221, 0.4530,
                                            0.4843, 0.5157, 0.5470, 0.5779, 0.6082, 0.6377, 0.6662,
                                            0.6935, 0.7196, 0.7442, 0.7674, 0.7891, 0.8092, 0.8279,
                                            0.8451, 0.8608, 0.8752, 0.8883, 0.9002, 0.9109, 0.9206,
                                            0.9293, 0.9371, 0.9441, 0.9504, 0.9560, 0.9610, 0.9654,
                                            0.9694, 0.9729, 0.9760, 0.9788, 0.9813, 0.9834, 0.9854,
                                            0.9871, 0.9886],
                'immunity_lut': True,

                # End Conditions
                'max_time': 31*24, # One month
                'max_intakes': None,

                # Intervention
                'intervention': 'TimedRemovalIntervention()' # Different interventions can go here
                }
        with open('./sim_params.json', 'w+') as out:
            json.dump(self.params, out)
            
        print(self.params['intervention'])
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state_from_simulation(self):
        states = []
        num_infected = 0
        for node in self.simulation.disease.graph.nodes:
            states.append(self.simulation.disease.graph.nodes[node]['data']['occupant']['state'])
            if states[-1] == 2:
                num_infected += 1
        return np.concatenate((np.array(FLATTEN(self.state_encoder.transform([[x, 1] for x in states]))), np.array([self.i, self.j, self.k, self.r]))), num_infected

    def _get_node_at_index(self, i):
        return list(self.simulation.disease.graph.nodes)[i]

    def _get_adjacent_edges(self, i):
        nodes_at_depth = [i]
        edges = [(start, end) for start, end in self.simulation.disease.graph.edges]
        d_edges = list(set([e[1] for e in edges if e[0] == i] + [e[0] for e in edges if e[1] == i]))
        return d_edges

    def _get_next_state(self, i, k):
        return self._get_adjacent_edges(i)[k]
        
    def _get_next_rotation(self, i, k):
        return (k + 1) % len(self._get_adjacent_edges(i))
    
    def _get_next_component(self, i):
        for idx, component in enumerate(self.components):
            if i in component:
                return self.components[(idx+1)%len(self.components)][0]

    def _get_prev_component(self, i):
        for idx, component in enumerate(self.components):
            if i in component:
                return self.components[(idx-1)%len(self.components)][0]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        _, num_infected = self._get_state_from_simulation()

        # Stats
        self.bonus_reward = 0.
        self.turn_around_counter += 1
        self.actions_history.append(action)
        update = False
        #reward = 0
        if action == 0:
            self.simulation.update()
            update = True
            if self.incentive_methods[0]:
                self.bonus_reward = 1.0 # Bonus reward method
            if self.incentive_methods[1]:
                self.turn_around_counter = 0 # Turn around rate method
            
        elif action == 1:
            #self.i = (self.i + 1) % self.num_nodes
            #if self.i == self.j:
            #    self.i = (self.i + 1) % self.num_nodes
            self.i = self._get_next_state(self.i, self.k)
            self.k = 0
        elif action == 2:
            #self.j = (self.j + 1) % self.num_nodes
            #if self.i == self.j:
            #    self.j = (self.i + self.j) % self.num_nodes
            self.j = self._get_next_state(self.j, self.r)
            self.r = 0
        elif action == 3:
            self.k = self._get_next_rotation(self.i, self.k)
        elif action == 4:
            self.r = self._get_next_rotation(self.j, self.r)
        elif action == 5:
            self.i = self._get_next_component(self.i)
            self.k = 0
        elif action == 6:
            self.j = self._get_next_component(self.j)
            self.r = 0
        elif action == 7:
            self.simulation.disease.swap_cells(self._get_node_at_index(self.i), 
                                               self._get_node_at_index(self.j))
            #reward = np.clip(self.max_expected_swaps-self.swaps_this_hour, -1, self.max_expected_swaps)
            #self.swaps_this_hour += 1
        
        if action != 0 and self.incentive_methods[1] and (self.turn_around_counter % self.turn_around_rate == 0):
            self.simulation.update()
            self.turn_around_counter = 0
            self.actions_history.append(-1) # Add force update indicator in the action history
            update = True
                                               
        new_state, new_num_infected = self._get_state_from_simulation()

        self.state = new_state

        done = self.simulation.disease.end_conditions()
        
        # infection_rate = 0.18 # Chance
        results = self.simulation._get_disease_stats()
        if results['E'] <= 0 or self.simulation.disease.time <= 24:
            reward = 0
        else:
            infection_rate = results['I'] / results['E']
            infection_rate_scaler = 100 # Higher means more gradual reward increases as infection rate approaches 0.0
            reward_scaler = 1000.0 # Scaling the magnitude of the reward (only impacts absolute magnitude of reward)
            a, b, c, d = [2.95735301, 46.32006702, -72.55387083, 98.66898389]
            infection_rate_chance = np.arctan((self.simulation.disease.time+d)*a)*b+c
            reward_offset = np.exp((1.0 - infection_rate_scaler)/(infection_rate_scaler*infection_rate_chance))
            reward = reward_scaler*(np.exp((1.0/infection_rate/infection_rate_scaler) - (1.0/infection_rate_chance))-reward_offset)
            reward = np.clip(reward, -10, 10)

        #infected_delta = (new_num_infected - num_infected)
        #if update:
            #self.swaps_this_hour = 0
            #if infected_delta <= 0:
                #reward = self.reward_bias
                #print('yay')
            #else:
                #reward = (float(self.reward_bias)/(float(infected_delta)+1.0)) + self.bonus_reward
                #print('ok')
                #reward *= 10
        
        return np.array(self.state), reward, done, {}

    def _get_action_stats(self):
        self.actions_history = np.asarray(self.actions_history)
        return {#'Hist': self.actions_history,
                'Len': len(self.actions_history),
                'Next': np.sum(self.actions_history==0),
                'Move i': np.sum(self.actions_history==1),
                'Move j': np.sum(self.actions_history==2),
                'Rotate i': np.sum(self.actions_history==3),
                'Rotate j': np.sum(self.actions_history==4),
                'Jump i to Next Component': np.sum(self.actions_history==5),
                'Jump j to Next Component': np.sum(self.actions_history==6),
                'Swap i and j': np.sum(self.actions_history==7),
                'Force Next': np.sum(self.actions_history==-1)
                }
                
    def _print_stat(self, **kwargs):
        
        action = kwargs.get('action')
        if action != None:
            print("Action #{}".format(action), end="\r")
        
    def reset(self, **kwargs):
        
        self._reset_params()
        self.simulation = Simulation(self.params,
                                     spatial_visualization=False,
                                     aggregate_visualization=False,
                                     return_on_equillibrium=True)
                                     
        self.reward_bias = 100.0 if kwargs.get('reward_bias') is None else kwargs.get('reward_bias')       
        self.bonus_reward = 0. if kwargs.get('bonus_reward') is None else kwargs.get('bonus_reward')
        self.turn_around_rate = 400 if kwargs.get('turn_around_rate') is None else kwargs.get('turn_around_rate') 
        self.turn_around_counter = 0
        
        # Action stats
        self.actions_history = []
        self.turn_around_actions_history = []

        self.i = self.start_i
        self.j = self.start_j
        self.k = 0
        self.r = 0
        self.state, _ = self._get_state_from_simulation()
        return np.array(self.state)

    def close(self):
        self.simulation.running = False
        
class Distemper2(Distemper):

    def __init__(self, **kwargs):
        super(Distemper2,self).__init__(**kwargs)
        
    # The Observation State now includes i,j information with shape (#nodes,#states+1)
    def _get_state_from_simulation(self):
        states = []
        num_infected = 0
        for node in self.simulation.disease.graph.nodes:
            states.append(self.simulation.disease.graph.nodes[node]['data']['occupant']['state'])
            if states[-1] == 2:
                num_infected += 1
        embedded_states = [[x, 0] for x in states]
        embedded_states[self.i][1],embedded_states[self.j][1] = 1,-1
        return FLATTEN(self.state_encoder.transform(embedded_states)), num_infected