"""This module contains the critical disease and kennel objects.
"""

import json
import logging
import sys
import copy

import matplotlib as mpl
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

import pygame # pylint: disable=E0401

FLATTEN = lambda l: [item for sublist in l for item in sublist]

class DistemperModel(object):
    '''This class models the canine distemper virus via a non-deterministic state machine
    operating on each node of a graph a la a cellular automata.
    '''

    def __init__(self, networkx_graph, params, reset=True):
        self.params = params
        self.graph = networkx_graph
        self.id_map = {'E': 0, 'S': 1, 'IS': 2, 'I': 3, 'SY': 4, 'D': 5}
        self.id_map_r = {0: 'E', 1: 'S', 2: 'IS', 3: 'I', 4: 'SY', 5: 'D'}
        if reset:
            self.reset()

    def reset(self):
        '''This function resets the module.
        '''

        self.time = 0
        self.total_intake = 0
        self.total_infected = 0
        self.total_died = 0
        self.total_discharged = 0
        self.state_graph = self.init_state_graph()
        
        self.E2I = 0
        self.sum_S2D_IS2D = 0
        self.E2S = 0
        self.E2IS = 0
        self.S2I = 0

    @staticmethod
    def copy(disease):
        '''Performs a deep copy of the disease model

        Arguments:
            disease {DistemperModel} -- the disease model to copy

        Returns:
            DistemperModel -- the copy of the disease model
        '''

        networkx_graph = copy.deepcopy(disease.graph)
        params = copy.deepcopy(disease.params)
        new_model = DistemperModel(networkx_graph, params)
        new_model.time = copy.deepcopy(disease.time) # pylint: disable=W0201
        total_intake = disease.total_intake + 1
        new_model.total_intake = total_intake - 1 # pylint: disable=W0201
        new_model.total_infected = copy.deepcopy(disease.total_infected) # pylint: disable=W0201
        new_model.total_died = copy.deepcopy(disease.total_died) # pylint: disable=W0201
        new_model.total_discharged = copy.deepcopy(disease.total_discharged) # pylint: disable=W0201
        new_model.state_graph = copy.deepcopy(disease.state_graph) # pylint: disable=W0201
        return new_model

    def change_node_state(self, node, old_state, new_state):
        '''This function changes a node's state, adjusting state tracking variables as well.

        Arguments:
            node {int} -- the node number
            old_state {int} -- the start node id integer (from id_map)
            new_state {int} -- the end node id integer (from id_map)
        '''

        if new_state == self.id_map['I'] or \
           (old_state == self.id_map['E'] and \
           new_state == self.id_map['SY']):
            self.total_infected += 1
        elif new_state == self.id_map['D']:
            self.total_died += 1
        elif old_state == self.id_map['IS'] and new_state == self.id_map['E']:
            self.total_discharged += 1
        node['occupant']['state'] = new_state
        if node['node_id'] in self.state_graph.nodes[old_state]['members']:
            self.state_graph.nodes[old_state]['members'].remove(node['node_id'])
        self.state_graph.nodes[new_state]['members'].append(node['node_id'])
        
        if old_state == self.id_map['E'] and new_state == self.id_map['I']:
            self.E2I += 1
        elif (old_state == self.id_map['S'] or old_state == self.id_map['IS']) and new_state == self.id_map['D']:
            self.sum_S2D_IS2D += 1
        elif old_state == self.id_map['E'] and new_state == self.id_map['S']:
            self.E2S += 1
        elif old_state == self.id_map['E'] and new_state == self.id_map['IS']:
            self.E2IS += 1
        elif old_state == self.id_map['S'] and new_state == self.id_map['I']:
            self.S2I += 1

    @staticmethod
    def get_occupant_element():
        '''Returns a default occupent element.

        Returns:
            dict -- a dictionary containing the default values for an occupant
        '''

        return {'value': 0,
                'state': 0,
                'immunity': 0,
                'intake_t': -1,
                'locked': False}

    def build_new_occupant(self, start_state, start_value=0, start_immunity=0.0):
        '''This function creates a new occupant for a node.

        Arguments:
            start_state {str} -- the node id string

        Keyword Arguments:
            start_value {int} -- the starting value (currently unused; default: {0})
            start_immunity {float} -- the starting immunity (default: {0.0})

        Returns:
            dict -- a dictionary of the occupant values
        '''

        self.total_intake += 1
        occupant_element = DistemperModel.get_occupant_element()
        occupant_element['value'] = start_value
        occupant_element['state'] = start_state
        occupant_element['immunity'] = start_immunity
        occupant_element['intake_t'] = self.time
        return occupant_element

    # pylint: disable=C0103
    def get_probability_parameter_with_refractory(self, node,
                                                  parameter_name=None,
                                                  refractory_parameter=None,
                                                  occupant_parameter=None):
        '''This function returns the probability of an event given a refractory period from intake.
        This function will return 0.0 if any Keyword input is None

        Arguments:
            node {int} -- the node id

        Keyword Arguments:
            parameter_name {str} -- the probability parameter name (default: {None})
            refractory_parameter {str} -- the refractory parameter name (default: {None})
            occupant_parameter {str} -- the occupant parameter to compare to current time
            (default: {None})

        Returns:
            float -- the probability of an event given the refractory period
        '''

        if parameter_name and refractory_parameter and occupant_parameter:
            if self.time - node['occupant'][occupant_parameter] > self.params[refractory_parameter]:
                return self.params[parameter_name]
        return 0.0

    def add_new_animal(self, node, state=None):
        '''Add a new animal to the simulation at a given node.
        If the starting state is None, nothing will happen in this function.

        Arguments:
            node {int} -- the node id

        Keyword Arguments:
            state {str} -- the starting state (default: {None})
        '''

        if state:
            node['occupant'] = self.build_new_occupant(self.id_map[state])
            self.change_node_state(node, self.id_map['E'], self.id_map[state])

    def get_infection_probability(self, node):
        '''Gets the probability of infection at a node given the kernel function and graph.

        Arguments:
            node {int} -- the node id of the target node

        Returns:
            float -- the probability of infection
        '''

        depth = len(self.params['infection_kernel'])
        kernel_function = self.params['infection_kernel_function']

        infection_kernel = np.clip([kernel_function(node, k) for
                                    k in self.params['infection_kernel']],
                                   0, 1)

        nodes_at_depth = [[node['node_id']]]
        nodes_at_depth_nearest = [[node['node_id']]]
        all_conn_nodes = [node['node_id']]
        edges = [(start, end) for start, end in self.graph.edges]
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

        infected_nodes = self.get_state_node('I')['members'] + self.get_state_node('SY')['members']

        probability_list = [] # Probability each event happens

        for depth, d_edges in enumerate(nodes_at_depth_nearest[1:]):
            for node_at_depth in d_edges:
                if node_at_depth in infected_nodes:
                    probability_list.append(infection_kernel[depth])

        return 1 - np.product([1-p for p in probability_list]) # Probability any event happens

    def update_susceptible(self, node):
        '''This function updates the immunity factor for susceptible animals.

        Arguments:
            node {int} -- the node id
        '''

        if node['occupant']['immunity'] < 1:
            if self.params['immunity_lut']:
                time = self.time - node['occupant']['intake_t']
                if time < 0:
                    node['occupant']['immunity'] = 0
                if time >= len(self.params['immunity_growth_factors']):
                    node['occupant']['immunity'] = 1
                else:
                    node['occupant']['immunity'] = self.params['immunity_growth_factors'][time]
            else:
                node['occupant']['immunity'] = node['occupant']['immunity'] * \
                                            self.params['immunity_growth_factors'][0] + \
                                            self.params['immunity_growth_factors'][1]


    def init_state_graph(self):
        '''This function initializes the state machine graph.

        Returns:
            nx.DiGraph -- the state machine graph representing the simulation logic
        '''

        state_graph = nx.DiGraph()

        all_nodes = [int(node) for node in self.graph.nodes]
        state_graph.add_node(0, node_id='E', name='Empty Cell',
                             update_function=None, members=all_nodes)
        state_graph.add_node(1, node_id='S', name='Susceptible Animal',
                             update_function=self.update_susceptible, members=[])
        state_graph.add_node(2, node_id='IS', name='Insusceptible Animal',
                             update_function=None, members=[])
        state_graph.add_node(3, node_id='I', name='Infected Animal',
                             update_function=None, members=[])
        state_graph.add_node(4, node_id='SY', name='Symptomatic',
                             update_function=None, members=[])
        state_graph.add_node(5, node_id='D', name='Deceased Animal',
                             update_function=None, members=[])

        assert self.params["pSusceptibleIntake"] + \
               self.params["pInsusceptibleIntake"] + \
               self.params["pInfectIntake"] + \
               self.params["pSymptomaticIntake"] <= 1.0, \
            "pSusceptibleIntake + pInsusceptibleIntake + pInfectIntake + " + \
            "pSymptomaticIntake must be less than 1.0"

        # New susceptible animal
        state_graph.add_edge(0, 1,
                             transition_criteria_function=lambda node:
                             self.params["pSusceptibleIntake"],
                             transition_function=lambda node: self.add_new_animal(node, 'S'))
        # New insusceptible animal
        state_graph.add_edge(0, 2,
                             transition_criteria_function=lambda node:
                             self.params["pInsusceptibleIntake"],
                             transition_function=lambda node: self.add_new_animal(node, 'IS'))
        # New infected animal
        state_graph.add_edge(0, 3,
                             transition_criteria_function=lambda node:
                             self.params["pInfectIntake"],
                             transition_function=lambda node: self.add_new_animal(node, 'I'))
        # New infected animal (with symptoms)
        state_graph.add_edge(0, 4,
                             transition_criteria_function=lambda node:
                             self.params["pSymptomaticIntake"],
                             transition_function=lambda node: self.add_new_animal(node, 'SY'))

        assert self.params['pDieAlternate'] <= 1.0, "pDieAlternate must be less than 1.0"

        # Animal gains immunity
        state_graph.add_edge(1, 2,
                             transition_criteria_function=lambda node:
                             int(node['occupant']['immunity'] >= 1) * \
                             (1 - self.params['pDieAlternate']),
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['S'],
                                                    self.id_map['IS']))
        # Animal becomes infected
        state_graph.add_edge(1, 3,
                             transition_criteria_function=self.get_infection_probability,
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['S'],
                                                    self.id_map['I']))
        # Susceptible animal dies from other causes
        state_graph.add_edge(1, 5,
                             transition_criteria_function=lambda node:
                             self.params["pDieAlternate"],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['S'],
                                                    self.id_map['D']))

        assert self.params['pDieAlternate'] + self.params['pDischarge'] <= 1.0, \
            "pDieAlternate + pDischarge must be less than 1.0"

        # Insusceptible animal is discharged
        state_graph.add_edge(2, 0,
                             transition_criteria_function=lambda node:
                             self.params['pDischarge'],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['IS'],
                                                    self.id_map['E']))
        # Insusceptible animal dies from other causes
        state_graph.add_edge(2, 5,
                             transition_criteria_function=lambda node:
                             self.params["pDieAlternate"],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['IS'],
                                                    self.id_map['D']))

        assert self.params['pSurviveInfected'] + \
               self.params['pSymptomatic'] + \
               self.params['pDieAlternate'] <= 1.0, \
            "pSurviveInfected + pSymptomatic + pDieAlternate must be less than 1.0"

        # Infected dog is discharged
        state_graph.add_edge(3, 2,
                             transition_criteria_function=lambda node:
                             self.get_probability_parameter_with_refractory(node,
                                                                            "pSurviveInfected",
                                                                            "refractoryPeriod",
                                                                            "intake_t"),
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['I'],
                                                    self.id_map['IS']))
        # Infected dog dies
        state_graph.add_edge(3, 4,
                             transition_criteria_function=lambda node:
                             self.get_probability_parameter_with_refractory(node,
                                                                            "pSymptomatic",
                                                                            "refractoryPeriod",
                                                                            "intake_t"),
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['I'],
                                                    self.id_map['SY']))
        # Infected dog dies from other causes
        state_graph.add_edge(2, 5,
                             transition_criteria_function=lambda node:
                             self.params["pDieAlternate"],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['I'],
                                                    self.id_map['D']))

        assert self.params['pSurviveSymptomatic'] + \
               self.params['pDie'] + \
               self.params["pDieAlternate"] <= 1.0, \
            "pSurviveSymptomatic + pDie + pDieAlternate must be less than 1.0"

        # Symptomatic dog is discharged
        state_graph.add_edge(4, 2, transition_criteria_function=lambda node:
                             self.get_probability_parameter_with_refractory(node,
                                                                            "pSurviveSymptomatic",
                                                                            "refractoryPeriod",
                                                                            "intake_t"),
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['SY'],
                                                    self.id_map['IS']))
        # Symptomatic dog dies
        state_graph.add_edge(4, 5, transition_criteria_function=lambda node:
                             self.get_probability_parameter_with_refractory(node,
                                                                            "pDie",
                                                                            "refractoryPeriod",
                                                                            "intake_t"),
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['SY'],
                                                    self.id_map['D']))
        # Symptomatic dog dies from other causes
        state_graph.add_edge(2, 5, transition_criteria_function=lambda node:
                             self.params["pDieAlternate"],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['SY'],
                                                    self.id_map['D']))

        assert self.params['pCleaning'] <= 1.0, "pCleaning must be less than 1.0"

        # Deceased dog kennel is emptied
        state_graph.add_edge(5, 0, transition_criteria_function=lambda node:
                             self.params['pCleaning'],
                             transition_function=lambda node:
                             self.change_node_state(node,
                                                    self.id_map['D'],
                                                    self.id_map['E']))

        return state_graph

    def get_state_node(self, state):
        '''Get the list of nodes in a given state.

        Arguments:
            state {str} -- the state string to get members

        Returns:
            list(int) -- a list of node ids that are in the state
        '''

        return self.state_graph.nodes[self.id_map[state]]

    def apply_state_graph(self):
        '''Apply the state graph to the kennel graph.
        '''

        for _, data in self.graph.nodes(data=True):
            node = data['data']
            # Skip if locked
            if node['occupant']['locked']:
                continue

            node_state = node['occupant']['state']
            current_state = self.state_graph.nodes[node_state]
            current_state_update_function = current_state['update_function']
            if current_state_update_function:
                current_state_update_function(node)
            edges = self.state_graph.edges(node_state, data=True)
            transition_criteria_functions = [edge_data['transition_criteria_function'] for start,
                                             end,
                                             edge_data in edges]
            transition_functions = [edge_data['transition_function'] for start,
                                    end,
                                    edge_data in edges]
            transitions_probabilities = [f(node) for f in transition_criteria_functions]
            if sum(transitions_probabilities) > 1:
                logging.error('Probabilities for transition sum to greater than 1.')
                sys.exit(1)
            null_event_p = 1 - sum(transitions_probabilities)
            full_probabilities = [null_event_p] + list(transitions_probabilities)
            transition = np.random.choice(list(range(0, len(full_probabilities))),
                                          1,
                                          p=full_probabilities)[0]

            if transition != 0:
                transition_functions[transition-1](node)

    @staticmethod
    def look_ahead(disease_state, n, sample=1, perform_first=None):
        '''This function creates a copy of the simulation as it is right now then
        iterates the simulation n times. It will perform this operation as many
        times as specified by sample then provide the list of results.

        Arguments:
            n {int} -- the number of steps to look ahead

        Keyword Arguments:
            sample {int} -- the number of times to try looking ahead (default: {1})

        Returns:
            list(list(float)) -- a list of the results (total intake, total infected)
            for each sample at time step t0+n where t0 is the current simulation state
        '''
        disease_copy = DistemperModel.copy(disease_state)
        results = []
        for _ in range(0, sample):
            disease = DistemperModel.copy(disease_copy)
            if perform_first:
                perform_first(disease)
            for _ in range(0, n):
                disease.update()
            results.append([disease.total_intake, disease.total_infected])
        return results

    def update(self):
        '''Update both the state graph and time.
        '''

        self.apply_state_graph()
        self.time += 1

    def in_equilibrium(self):
        '''Check if the simulation is in equillibrium (all cages in stable states)

        Returns:
            bool -- True if no empty, susceptible, infected, or symptomatic cages
        '''

        empty_nodes = self.get_state_node('E')['members']
        susceptible_nodes = self.get_state_node('S')['members']
        infected_nodes = self.get_state_node('I')['members']
        symptomatic_nodes = self.get_state_node('SY')['members']

        return len(susceptible_nodes) == 0 and \
               len(empty_nodes) == 0 and \
               len(infected_nodes) == 0 and \
               len(symptomatic_nodes) == 0

    def end_conditions(self):
        '''Check if end conditions are met. This is an alternative to equillibrium conditions.

        Returns:
            bool -- True of max_time or max_intakes is reached
            (they are ignored if None or not present)
        '''

        if 'max_time' in self.params and \
            self.params['max_time'] and \
            self.params['max_time'] <= self.time:
            return True
        if 'max_intakes' in self.params and \
            self.params['max_intakes'] and \
            self.params['max_intakes'] <= self.total_intake:
            return True
        return False

    def swap_cells(self, node_id0, node_id1):
        '''Swap two cell contents

        Arguments:
            node_id0 {int} -- the first node id
            node_id1 {int} -- the second node id
        '''

        node0 = self.graph.nodes[node_id0]
        node1 = self.graph.nodes[node_id1]
        state0 = node0['data']['occupant']['state']
        state1 = node1['data']['occupant']['state']
        # Swap state membership
        if node_id0 in self.state_graph.nodes[state0]['members']:
            self.state_graph.nodes[state0]['members'].remove(node_id0)
        self.state_graph.nodes[state1]['members'].append(node_id0)
        if node_id1 in self.state_graph.nodes[state1]['members']:
            self.state_graph.nodes[state1]['members'].remove(node_id1)
        self.state_graph.nodes[state0]['members'].append(node_id1)
        # Swap occupant
        tmp = node1['data']['occupant']
        node1['data']['occupant'] = node0['data']['occupant']
        node0['data']['occupant'] = tmp


class Kennels(object):
    '''This object represents the kennel graph and its associated rendering.
    '''

    def __init__(self, kennel_layout_definition_file=None,
                 colors=None,
                 edge_color=(255, 0, 0),
                 background_color=(255, 255, 255)):
        if colors is None:
            colors = {'E': (0, 0, 0),
                      'S': (0, 0, 255),
                      'IS': (0, 255, 0),
                      'I': (255, 255, 0),
                      'SY': (255, 165, 0),
                      'D': (255, 0, 0)}
        self.colors = colors
        self.edge_color = edge_color
        self.background_color = background_color
        convert_color_to_01 = lambda color_tuple: tuple(np.array(color_tuple)/255)
        convert_color_to_0255 = lambda color_tuple: tuple([int(x*255) for x in color_tuple])
        immunity_colormap = \
            mpl.colors.LinearSegmentedColormap.from_list('immunity_colormap',
                                                         [convert_color_to_01(self.colors['S']),
                                                          convert_color_to_01(self.colors['IS'])],
                                                         256)
        self.immunity_gradient = [convert_color_to_0255(immunity_colormap(x)[0:3])
                                  for x in np.linspace(0, 1, 256)]
        if kennel_layout_definition_file is None:
            self.graph = Kennels.get_sample_kennel_graph()
            self.save_to_files('./data/test.graph')
        else:
            self.load_from_files(kennel_layout_definition_file)


    @staticmethod
    def get_sample_kennel_graph(grid=None,
                                graphics_params=None):
        '''This function generates a sample kennel graph in a grid.

        Keyword Arguments:
            grid {list(list(int))} -- a list of cage row sizes in disconnected groups
            (default: {None; [[12, 12], [12, 12], [0], [12, 12], [12, 12]]})
            graphics_params {dict(int)} -- a dictionary containing node_size, node_minor_pad,
                                        node_major_pad, x_offset, and y_offset for
                                        visualization (default: {None;
                                        {'node_size': 10, 'node_minor_pad': 1,
                                         'node_major_pad': 5,
                                         'x_offset': 50, 'y_offset': 50}})
        '''

        if grid is None:
            grid = [[12, 12], [12, 12], [0], [12, 12], [12, 12]]
        if graphics_params is None:
            graphics_params = {'node_size': 10, 'node_minor_pad': 1,
                               'node_major_pad': 5,
                               'x_offset': 50, 'y_offset': 50,
                               'immunity_gradient': True}
        nodes = []
        edges = []
        count = 0
        row_offset = 0
        for row in grid:
            col_offset = 0
            for segment_length in row:
                for i in range(0, segment_length):
                    new_node = {
                        'node_id': count,
                        'x': col_offset + graphics_params['x_offset'],
                        'y': row_offset + graphics_params['y_offset'],
                        'color': (0, 0, 0),
                        'occupant': DistemperModel.get_occupant_element()
                    }
                    new_node['center'] = Kennels.get_nodes_center(new_node,
                                                                  graphics_params['node_size'])
                    nodes.append(new_node)
                    if i != segment_length - 1:
                        edges.append({'start': count, 'end': count + 1})
                    count += 1
                    col_offset += graphics_params['node_minor_pad'] + graphics_params['node_size']
                col_offset += graphics_params['node_major_pad']
            row_offset += graphics_params['node_major_pad'] + graphics_params['node_size']
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node['node_id'], data=node)
        for edge in edges:
            graph.add_edge(edge['start'], edge['end'])

        graph.graphics_params = graphics_params
        return graph

    def load_from_files(self, filepath):
        '''Load a graph from file.

        Arguments:
            filepath {str} -- the path to the graph file
        '''

        with open(filepath, 'r') as file_pointer:
            data = file_pointer.read()
            data = json.loads(data)
            graphics_params = data.pop('graphics_params', None)
            self.graph = json_graph.node_link_graph(data)
            self.graph.graphics_params = graphics_params


    def save_to_files(self, filepath, indent=1):
        '''Save the graph to file.

        Arguments:
            filepath {str} -- the path to save the file

        Keyword Arguments:
            indent {int} -- the indentation level to pass to json.dump (default: {1})
        '''

        data = nx.node_link_data(self.graph)
        data['graphics_params'] = self.graph.graphics_params
        with open(filepath, 'w') as file_pointer:
            json.dump(data, file_pointer, indent=indent)

    def get_graph(self):
        '''Get the graph.

        Returns:
            nx.Graph -- the kennel graph
        '''

        return self.graph

    def set_graph(self, graph):
        '''Set the graph.

        Arguments:
            graph {nx.Graph} -- the kennel graph
        '''

        self.graph = graph

    @staticmethod
    def draw_box(surf, color, pos, size):
        '''Draw a box in the game window.

        Arguments:
            surf {Surface} -- the surface to draw to
            color {color} -- the color to draw
            pos {list(int, int)} -- the position to draw to (top left)
            size {list(int, int)} -- the size to draw
        '''

        rect = pygame.Rect((pos[0], pos[1]), (size[0], size[1]))
        pygame.draw.rect(surf, color, rect)

    @staticmethod
    def draw_line(surf, color, pos0, pos1):
        '''Draw a line in the game window.

        Arguments:
            surf {Surface} -- the surface to draw to
            color {color} -- the color to draw
            pos0 {list(int, int)} -- the start position of the line
            pos1 {list(int, int)} -- the end position of the line
        '''

        pygame.draw.line(surf, color, pos0, pos1)

    @staticmethod
    def get_nodes_center(node, size):
        '''Get the center of a drawn node.

        Arguments:
            node {int} -- the node id
            size {list(int, int)} -- the size of the node

        Returns:
            tuple(int, int) -- the center of the node
        '''

        return (node['x']+size/2.0, node['y']+size/2.0)

    def draw(self, surf, disease):
        '''Draw the kennel given a disease state.

        Arguments:
            surf {Surface} -- the draw surface
            disease {DistemperModel} -- the current disease state
        '''

        empty_nodes = disease.get_state_node('E')['members']
        susceptible_nodes = disease.get_state_node('S')['members']
        survived_nodes = disease.get_state_node('IS')['members']
        infected_nodes = disease.get_state_node('I')['members']
        symptomatic_nodes = disease.get_state_node('SY')['members']
        died_nodes = disease.get_state_node('D')['members']

        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['data']
            color = node['color']
            if node['node_id'] in susceptible_nodes:
                if self.graph.graphics_params['immunity_gradient']:
                    color_idx = int(node['occupant']['immunity']*float(255))
                    color = self.immunity_gradient[color_idx]
                else:
                    color = self.colors['S']
            elif node['node_id'] in empty_nodes:
                color = self.colors['E']
            elif node['node_id'] in infected_nodes:
                color = self.colors['I']
            elif node['node_id'] in survived_nodes:
                color = self.colors['IS']
            elif node['node_id'] in symptomatic_nodes:
                color = self.colors['SY']
            elif node['node_id'] in died_nodes:
                color = self.colors['D']
            Kennels.draw_box(surf, color, (node['x'], node['y']),
                             [self.graph.graphics_params['node_size'],
                              self.graph.graphics_params['node_size']])
        for edge in self.graph.edges:
            Kennels.draw_line(surf, self.edge_color,
                              self.graph.nodes[edge[0]]['data']['center'],
                              self.graph.nodes[edge[1]]['data']['center'])

if __name__ == '__main__':
    from main import main
    main()
