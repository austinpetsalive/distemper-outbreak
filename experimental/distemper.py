"""
Distemper simulation for RL Agent.
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from simulation import Simulation

from sklearn.preprocessing import OneHotEncoder

FLATTEN = lambda l: [item for sublist in l for item in sublist]

class Distemper(gym.Env):
    """
    Description:
        A simulation of the canine distemper virus.

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Number of Infected          0           Inf
        1   Kenne States                ?           ?
        2	Number of Intakes           0           Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Move to next iteration
        1	Increment i
        2	Increment j
        3   Swap i and j contents

    Reward:
        10-(Number of Infected)

    Starting State:
        Empty Kennel Layout

    Episode Termination:
        31 Days (744 simulation ticks/hours)
        Consider early stopping if infection ratio is too high (Greater than 0.5)
    """

    def __init__(self):
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

        self.simulation = Simulation(self.params,
                                     spatial_visualization=False,
                                     aggregate_visualization=False,
                                     return_on_equillibrium=True)
        self.state_encoder = OneHotEncoder(handle_unknown='error', sparse=False)
        self.state_encoder.fit([[x, 1] for x in self.simulation.disease.id_map.values()])
        self.num_nodes = len(self.simulation.disease.graph.nodes)
        self.num_states = len(self.simulation.disease.id_map.values())
        self.action_space = spaces.Discrete(4)
        self.num_states = len(self.state_encoder.transform([[0, 1]])[0])
        self.observation_space = spaces.Discrete(self.num_states*self.num_nodes)

        self.reward_bias = 5.0
        
        # Randomly initialize i and j
        self.start_i = np.random.randint(0, self.num_nodes)
        self.start_j = np.random.randint(0, self.num_nodes)
        # If they happen to be equal, adjust j randomly up or down 1 (circling around to 0 if needed)
        if self.start_i == self.start_j:
            adjustment = -1 if bool(np.random.randint(2)) else 1
            self.start_j = (self.start_j + adjustment) % self.num_nodes
        self.i = self.start_i
        self.j = self.start_j

        # Rotation
        self.k = 0
        self.r = 0

        self.seed()

        self.state, _ = self._get_state_from_simulation()

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
        return FLATTEN(self.state_encoder.transform([[x, 1] for x in states])), num_infected

    def _get_node_at_index(self, i):
        return list(self.simulation.disease.graph.nodes)[i]

    def _get_next_state(self, i, k):
        i_node = _get_node_at_index(i)

    def _get_next_rotation(self, k):
        i_node = _get_node_at_index(i)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        _, num_infected = self._get_state_from_simulation()

        if action == 0:
            self.simulation.update()
        elif action == 1:
            self.i = (self.i + 1) % self.num_nodes
            if self.i == self.j:
                self.i = (self.i + 1) % self.num_nodes
        elif action == 2:
            self.j = (self.j + 1) % self.num_nodes
            if self.i == self.j:
                self.j = (self.i + self.j) % self.num_nodes
        elif action == 3:
            self.simulation.disease.swap_cells(self._get_node_at_index(self.i),
                                               self._get_node_at_index(self.j))

        new_state, new_num_infected = self._get_state_from_simulation()

        self.state = new_state

        done = self.simulation.disease.end_conditions()
        if action == 0:
            reward = self.reward_bias - (new_num_infected - num_infected)
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.simulation = Simulation(self.params,
                                     spatial_visualization=False,
                                     aggregate_visualization=False,
                                     return_on_equillibrium=True)
        self.i = 0
        self.j = 0
        self.state, _ = self._get_state_from_simulation()
        return np.array(self.state)

    def close(self):
        self.simulation.running = False
