import pickle as pkl
import networkx as nx
import numpy as np
import pygame
import random
import time
import json

from networkx.readwrite import json_graph

flatten = lambda l: [item for sublist in l for item in sublist]

class DistemperModel(object):
    def __init__(self, networkx_graph, params):
        self.params = params
        self.graph = networkx_graph
        self.transition_conflict_policy = 'random' # 'first' for first on state list
        self.reset()

    def reset(self):
        self.t = 0
        self.state_graph = self.init_state_graph()

    def change_node_state(self, node, old_state, new_state):
        node['occupant']['state'] = new_state
        if node['node_id'] in self.state_graph.nodes[old_state]['members']:
            self.state_graph.nodes[old_state]['members'].remove(node['node_id'])
        self.state_graph.nodes[new_state]['members'].append(node['node_id'])

    def build_new_occupant(self, start_state, start_value=0, start_immunity=0.0):
        return {'value': start_value, 'state': start_state, 'immunity': start_immunity, 'intake_t': self.t}

    def susceptible_intake_allocated(self, node):
        return random.random() < self.params['pIntake']

    def add_susceptible_animal(self, node):
        node['occupant'] = self.build_new_occupant(1)
        self.change_node_state(node, 0, 1)

    def insusceptible_intake_allocated(self, node):
        return random.random() < 0.0

    def add_insusceptible_animal(self, node):
        node['occupant'] = self.build_new_occupant(2)
        self.change_node_state(node, 0, 2)

    def infected_intake_allocated(self, node):
        return random.random() < self.params['pInfect']

    def add_infected_animal(self, node):
        node['occupant'] = self.build_new_occupant(3)
        self.change_node_state(node, 0, 3)

    def check_immunity_condition(self, node):
        if node['occupant']['immunity'] >= 1:
            return True
        else:
            return False
    
    def make_immune(self, node):
        self.change_node_state(node, 1, 2)

    def check_infection_condition(self, node):
        depth = len(self.params['infection_kernel'])
        kernel_function = self.params['infection_kernel_function']

        infection_kernel = np.clip([kernel_function(node, k) for k in self.params['infection_kernel']], 0, 1)

        nodes_at_depth = [[node['node_id']]]
        nodes_at_depth_nearest = [[node['node_id']]]
        all_conn_nodes = [node['node_id']]
        edges = [(start, end) for start, end in self.graph.edges]
        for d in range(1, depth + 1):
            prev_depth = nodes_at_depth[-1] # Store previous depth nodes
            d_edges = list(set(flatten([[e[1] for e in edges if e[0] == e0] for e0 in prev_depth]) + flatten([[e[0] for e in edges if e[1] == e0] for e0 in prev_depth]))) # Get connected nodes
            nodes_at_depth.append(d_edges) # Add nodes at this depth to depth list
            nodes_at_depth_nearest.append(list(set(d_edges) - set(all_conn_nodes))) # Add nodes to nearest only if they aren't already included
            all_conn_nodes.extend(nodes_at_depth_nearest[-1]) # Populate new nodes that have been added on inventory
        
        infected_nodes = self.get_state_node('I')['members']
        
        infected = False
        for depth, d_edges in enumerate(nodes_at_depth_nearest[1:]):
            for d in d_edges:
                if d in infected_nodes:
                    if random.random() < infection_kernel[depth]:
                        infected = True
                        break
        
        return infected
    
    def make_infected(self, node):
        self.change_node_state(node, 1, 3)

    def check_susceptible_death(self, node):
        return random.random() < self.params['pDieAlternate']

    def make_susceptible_deceased(self, node):
        self.change_node_state(node, 1, 4)

    def check_insusceptible_death(self, node):
        return random.random() < self.params['pDieAlternate']

    def make_insusceptible_deceased(self, node):
        self.change_node_state(node, 2, 4)

    def check_infected_cured(self, node):
        if self.t - node['occupant']['intake_t'] > self.params['refractoryPeriod']:
            return random.random() < self.params['pSurvive']

    def make_infected_cured(self, node):
        self.change_node_state(node, 3, 2)

    def check_infected_death(self, node):
        if self.t - node['occupant']['intake_t'] > self.params['refractoryPeriod']:
            return random.random() < self.params['pDie']

    def make_infected_deceased(self, node):
        self.change_node_state(node, 3, 4)

    def update_empty(self, node):
        return
    
    def update_susceptible(self, node):
        if node['occupant']['immunity'] < 1:
            # equation for 5 day full immunity
            node['occupant']['immunity'] = node['occupant']['immunity']*self.params['immunity_growth_factors'][0] + self.params['immunity_growth_factors'][1]

    def update_insusceptible(self, node):
        return

    def update_infected(self, node):
        return

    def update_deceased(self, node):
        return

    def init_state_graph(self):
        sG = nx.DiGraph()
        self.id_map = {'E': 0, 'S': 1, 'IS': 2, 'I': 3, 'D': 4}
        all_nodes = [int(node) for node in self.graph.nodes]
        sG.add_node(0, node_id='E', name='Empty Cell', update_function=self.update_empty, members=all_nodes)
        sG.add_node(1, node_id='S', name='Susceptible Animal', update_function=self.update_susceptible, members=[])
        sG.add_node(2, node_id='IS', name='Insusceptible Animal', update_function=self.update_insusceptible, members=[])
        sG.add_node(3, node_id='I', name='Infected Animal', update_function=self.update_infected, members=[])
        sG.add_node(4, node_id='D', name='Deceased Animal', update_function=self.update_deceased, members=[])
        
        sG.add_edge(0, 1, transition_criteria_function=self.susceptible_intake_allocated, transition_function=self.add_susceptible_animal) # New susceptible animal
        sG.add_edge(0, 2, transition_criteria_function=self.insusceptible_intake_allocated, transition_function=self.add_insusceptible_animal) # New insusceptible animal
        sG.add_edge(0, 3, transition_criteria_function=self.infected_intake_allocated, transition_function=self.add_infected_animal) # New infected animal
        
        sG.add_edge(1, 2, transition_criteria_function=self.check_immunity_condition, transition_function=self.make_immune) # Animal gains immunity
        sG.add_edge(1, 3, transition_criteria_function=self.check_infection_condition, transition_function=self.make_infected) # Animal becomes infected
        sG.add_edge(1, 4, transition_criteria_function=self.check_susceptible_death, transition_function=self.make_susceptible_deceased) # Susceptible animal dies from other causes

        sG.add_edge(2, 4, transition_criteria_function=self.check_insusceptible_death, transition_function=self.make_insusceptible_deceased) # Insusceptible animal dies from other causes

        sG.add_edge(3, 2, transition_criteria_function=self.check_infected_cured, transition_function=self.make_infected_cured) # Infected dog dies
        sG.add_edge(3, 4, transition_criteria_function=self.check_infected_death, transition_function=self.make_infected_deceased) # Infected dog dies

        return sG

    def get_state_node(self, node_id):
        return self.state_graph.nodes[self.id_map[node_id]]

    def apply_state_graph(self):
        for _, data in self.graph.nodes(data=True):
            node = data['data']
            if node['occupant'] is None:
                node_state = 0
            else:
                node_state = node['occupant']['state']
            current_state = self.state_graph.nodes[node_state]
            current_state_update_function = current_state['update_function']
            if current_state_update_function:
                current_state_update_function(node)
            edges = self.state_graph.edges(node_state, data=True)
            transition_criteria_functions = [edge_data['transition_criteria_function'] for start, end, edge_data in edges]
            transition_functions = [edge_data['transition_function'] for start, end, edge_data in edges]
            transitions = [f(node) for f in transition_criteria_functions]
            valid_true = transitions.count(True)
            f = lambda x: None
            if valid_true == 0:
                continue
            elif valid_true == 1:
                f = transition_functions[transitions.index(True)]
            else:
                if self.transition_conflict_policy == 'first':
                    f = transition_functions[transitions.index(True)]
                elif self.transition_conflict_policy == 'random':
                    idxs = [x for x, v in enumerate(transitions) if v]
                    choice = random.choice(idxs)
                    f = transition_functions[choice]

            f(node)
            
    def update(self, kennel):
        self.apply_state_graph()
        self.t += 1

    def in_equilibrium(self):
        empty_nodes = self.get_state_node('E')['members']
        susceptible_nodes = self.get_state_node('S')['members']
        # survived_nodes = self.get_state_node('IS')['members']
        infected_nodes = self.get_state_node('I')['members']
        # died_nodes = self.get_state_node('D')['members']
        
        return len(susceptible_nodes) == 0 and len(empty_nodes) == 0 and len(infected_nodes) == 0

    def swap_cells(self, node_id0, node_id1):
        n0 = self.graph.nodes[node_id0]
        n1 = self.graph.nodes[node_id1]
        if n0['data']['occupant'] is None:
            s0 = 0
        else:
            s0 = n0['data']['occupant']['state']
        if n1['data']['occupant'] is None:
            s1 = 0
        else:
            s1 = n1['data']['occupant']['state']
        # Swap state membership
        self.state_graph.nodes[s0]['members'].remove(node_id0)
        self.state_graph.nodes[s1]['members'].append(node_id0)
        self.state_graph.nodes[s1]['members'].remove(node_id1)
        self.state_graph.nodes[s0]['members'].append(node_id1)
        # Swap occupant
        tmp = n1['data']['occupant']
        n1['data']['occupant'] = n0['data']['occupant']
        n0['data']['occupant'] = tmp


class Kennels(object):
    def __init__(self, kennel_layout_definition_file=None):
        if kennel_layout_definition_file is None:
            self.G = Kennels.get_sample_kennel_graph()
            self.save_to_files('test.graph')
        else:
            self.load_from_files(kennel_layout_definition_file)

        
    @staticmethod
    def get_sample_kennel_graph(grid=[[12, 12], [12, 12], [0], [12, 12], [12, 12]], 
                                graphics_params={'node_size': 10, 'node_minor_pad': 1, 
                                'node_major_pad': 5, 'x_offset': 50, 'y_offset': 50}):
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
                        'occupant': None
                        }
                    new_node['center'] = Kennels.get_nodes_center(new_node, graphics_params['node_size'])
                    nodes.append(new_node)
                    if i != segment_length - 1:
                        edges.append({'start': count, 'end': count + 1})
                    count += 1
                    col_offset += graphics_params['node_minor_pad'] + graphics_params['node_size']
                col_offset += graphics_params['node_major_pad']
            row_offset += graphics_params['node_major_pad'] + graphics_params['node_size']
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['node_id'], data=node)
        for edge in edges:
            G.add_edge(edge['start'], edge['end'])

        G.graphics_params = graphics_params
        return G

    def load_from_files(self, filepath):
        with open(filepath, 'r') as fp:
            data = fp.read()
            data = json.loads(data)
            graphics_params = data.pop('graphics_params', None)
            self.G = json_graph.node_link_graph(data)
            self.G.graphics_params = graphics_params
            

    def save_to_files(self, filepath, indent=1):
        data = nx.node_link_data(self.G)
        data['graphics_params'] = self.G.graphics_params
        with open(filepath, 'w') as fp:
            json.dump(data, fp, indent=indent)

    def get_graph(self):
        return self.G

    def set_graph(self, G):
        self.G = G

    @staticmethod
    def draw_box(surf, color, pos, size):
        r = pygame.Rect((pos[0], pos[1]), (size[0], size[1]))
        pygame.draw.rect(surf, color, r)

    @staticmethod
    def draw_line(surf, color, pos0, pos1, width=1):
        pygame.draw.line(surf, color, pos0, pos1)

    @staticmethod
    def get_nodes_center(node, size):
        return (node['x']+size/2.0, node['y']+size/2.0)

    def draw(self, surf, disease):
        empty_nodes = disease.get_state_node('E')['members']
        susceptible_nodes = disease.get_state_node('S')['members']
        survived_nodes = disease.get_state_node('IS')['members']
        infected_nodes = disease.get_state_node('I')['members']
        died_nodes = disease.get_state_node('D')['members']
        
        for node_id in self.G.nodes:
            node = self.G.nodes[node_id]['data']
            color = node['color']
            if node['node_id'] in susceptible_nodes:
                color = (0, 0, 255)
            elif node['node_id'] in empty_nodes:
                color = (0, 0, 0)
            elif node['node_id'] in infected_nodes:
                color = (255, 255, 0)
            elif node['node_id'] in survived_nodes:
                color = (0, 255, 0)
            elif node['node_id'] in died_nodes:
                color = (255, 0, 0)
            Kennels.draw_box(surf, color, (node['x'], node['y']), [self.G.graphics_params['node_size'], self.G.graphics_params['node_size']])
        for edge in self.G.edges:
            Kennels.draw_line(surf, (255, 0, 0), self.G.nodes[edge[0]]['data']['center'], self.G.nodes[edge[1]]['data']['center'])

if __name__ == '__main__':
    from main import main
    main()