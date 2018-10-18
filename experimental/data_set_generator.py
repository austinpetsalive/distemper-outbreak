# pylint: disable=C0111
import itertools
import random

import numpy as np

import simulation

from custom_disease_model import DistemperModel

PARAMS = {
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

SIM = simulation.Simulation(PARAMS,
                            spatial_visualization=False,
                            return_on_equillibrium=True,
                            aggregate_visualization=False)

FLATTEN = lambda l: [item for sublist in l for item in sublist]

def node_feature(sim, node, depth=2):
    # Compute nodes at distances in neighborhood
    disease = sim.disease
    edges = list(disease.graph.edges)
    nodes_at_depth = [[node]]
    nodes_at_depth_nearest = [[node]]
    all_conn_nodes = [node]
    edges = [(start, end) for start, end in disease.graph.edges]
    for _ in range(1, depth + 1):
        # Store previous depth nodes
        prev_depth = nodes_at_depth[-1]
        # Get connected nodes
        d_edges = list(set(FLATTEN([[e[1] for e in edges if e[0] == e0] for e0 in prev_depth]) +
                           FLATTEN([[e[0] for e in edges if e[1] == e0] for e0 in prev_depth])))
        # Add nodes at this depth to depth list
        nodes_at_depth.append(d_edges)
        # Add nodes to nearest only if they aren't already included
        nodes_at_depth_nearest.append(list(set(d_edges) - set(all_conn_nodes)))
        # Populate new nodes that have been added on inventory
        all_conn_nodes.extend(nodes_at_depth_nearest[-1])
    # Compute states of nodes at depths
    states_at_depth = []
    for nodes in nodes_at_depth_nearest[1:]:
        states = []
        for current_node in nodes:
            states.append(disease.graph.nodes[current_node]['data']['occupant']['state'])
        states_at_depth.append(states)
    features_at_depth = []
    for nodes in states_at_depth:
        features_at_depth.append([nodes.count(state) for
                                  state in list(range(0, len(disease.id_map)))])
    target_node_state = disease.graph.nodes[node]['data']['occupant']['state']
    full_features = [target_node_state] + FLATTEN(features_at_depth)
    return full_features

def build_full_feature_map(sim, depth=2, subsample=0.5, look_ahead_distance=1, look_ahead_sample=1):
    all_features = []
    all_results = []

    node_features = []
    for node in sim.disease.graph.nodes:
        node_features.append(node_feature(sim, node, depth=depth))
    swap_indexes = list(itertools.combinations(range(0, len(node_features)), 2))
    samples = random.sample(range(0, len(swap_indexes)), int(len(swap_indexes)*subsample))

    def perform_first(_disease, _i0, _i1):
        _disease.swap_cells(_i0, _i1)

    for sample in samples:
        results = DistemperModel.look_ahead(sim.disease, look_ahead_distance,
                                            sample=look_ahead_sample,
                                            perform_first=
                                            lambda x: perform_first(x,
                                                                    swap_indexes[sample][0], # pylint: disable=W0640
                                                                    swap_indexes[sample][1])) # pylint: disable=W0640
        average_results = np.mean(results, axis=0)
        features = node_features[swap_indexes[sample][0]] + node_features[swap_indexes[sample][1]]
        all_features.append(features)
        all_results.append(average_results)

    return all_features, all_results



X = []
Y = []

for i in range(0, 744):
    _X, _Y = build_full_feature_map(SIM)
    X.extend(_X)
    Y.extend(_Y)
    SIM.update()
