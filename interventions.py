"""This module contains the intervention objects.
"""

import networkx as nx

FLATTEN = lambda l: [item for sublist in l for item in sublist]

class SortIntervention(object):
    '''An intervention that sorts the kennel graph by immunity/susceptibility.
    '''

    def __init__(self, name=''):
        self.name = name
        self.swap_count = 0

    def update(self, simulation):
        '''Update the intervention given the simulation state.

        Arguments:
            simulation {Simulation} -- the current simulation state

        Returns:
            int -- the number of swaps updated
        '''

        swap_list = self._compute_swap_map(simulation.disease.graph)
        for swap in swap_list:
            self.swap_count += 1
            simulation.disease.swap_cells(swap[0], swap[1])
        return self.swap_count

    def _compute_swap_map(self, graph):
        swap_list = []
        node_weights = []
        for node in graph.nodes:
            data = graph.nodes[node]['data']
            if data['occupant']['state'] == 4: # Infected and Symptomatic
                node_weights.append(2)
            elif data['occupant']['state'] == 2 or \
                 data['occupant']['state'] == 0 or \
                 data['occupant']['state'] == 5: # Not susceptible/empty/died
                node_weights.append(1)
            else:
                node_weights.append(data['occupant']['immunity'])
        weighted_nodes = list(zip(list(range(0, len(node_weights))), node_weights))

        # Node, this does NOT compute the most efficient swap path
        for j, _ in enumerate(weighted_nodes):
            i_max = -1
            v_max = -1
            for i in range(j, len(weighted_nodes)):
                if weighted_nodes[i][1] > v_max:
                    v_max = weighted_nodes[i][1]
                    i_max = i

            if i_max != j:
                tmp = weighted_nodes[j]
                weighted_nodes[j] = weighted_nodes[i_max]
                weighted_nodes[i_max] = tmp
                swap_list.append([j, i_max])

        return swap_list

def move_out(simulation, node):
    '''This function should move the cell contents out to an unconnected node.
    If an empty unconnected node exists, it can go there, if not, a new one should be created.
    This function should apply an empty state to the original source node

    Arguments:
        simulation {Simulation} -- the simulation to modify
        node {int} -- the node to move out
    '''

    current_state = simulation.disease.graph.nodes[node]['data']['occupant']['state']
    simulation.disease.change_node_state(simulation.disease.graph.nodes[node]['data'],
                                         current_state,
                                         simulation.disease.id_map['E'])

class TimedRemovalIntervention(object):
    '''An intervention where no animals are ever moved, but all animals are removed
    from the graph after 72 hours, after the compartments are all full, or after
    becoming symptomatic. Animals are removed to an independent area.
    '''
    def __init__(self, name='', leave_time=72):
        self.name = name
        self.swap_count = 0
        self.leave_time = leave_time

    def update(self, simulation): #pylint: disable=W0613
        '''Update the intervention given the simulation state.

        Arguments:
            simulation {Simulation} -- the current simulation state

        Returns:
            int -- the number of swaps updated
        '''
        get_node_state = lambda node: \
            simulation.disease.graph.nodes[node]['data']['occupant']['state']
        get_node_intake = lambda node: \
            simulation.disease.graph.nodes[node]['data']['occupant']['intake_t']
        get_state_from_id = lambda id: \
            simulation.disease.id_map[id]

        for node in simulation.disease.graph.nodes:
            if simulation.disease.time - get_node_intake(node) > self.leave_time:
                move_out(simulation, node)
            elif get_node_state(node) == get_state_from_id('SY'):
                move_out(simulation, node)

        return self.swap_count

class RoomLockIntervention(object):
    '''An intervention that finds the connected components in a graph and every 24 hours,
    moves the batch of animals between the connected components. At the end of 72 hours,
    the animals are moved to an independent area. If an animal becomes symptomatic,
    they are also removed to an independent area.

    Note: This intervention assumes that the connected kennel components are equal in size.
    '''

    def __init__(self, name='', number_of_batches=4, switch_time=18):
        self.name = name
        self.swap_count = 0
        self.first_update = True
        self.components = []
        self.number_of_batches = number_of_batches
        self.current_batch = 0
        self.switch_time = switch_time
        self.orphan_nodes = []
        self.batches = [[]]

    def update(self, simulation): #pylint: disable=W0613
        '''Update the intervention given the simulation state.

        Arguments:
            simulation {Simulation} -- the current simulation state

        Returns:
            int -- the number of swaps updated
        '''
        if self.first_update:
            self.first_update = False
            self.components = [list(x) for x in nx.connected_components(simulation.disease.graph)]
            n_components = len(self.components)
            assert n_components >= self.number_of_batches, \
                ("there are not enough independent components in the "
                 "kennel graph to satisfy the requirement of {0} batches "
                 "({1} components found)").format(self.number_of_batches, n_components)

            n_components_per_batch = int(float(n_components)/float(self.number_of_batches))
            batches = [[]]
            for component in self.components:
                if len(batches[-1]) < n_components_per_batch:
                    batches[-1].append(component)
                else:
                    if len(batches) == self.number_of_batches:
                        break
                    batches.append([component])
            for idx, batch in enumerate(batches):
                batches[idx] = FLATTEN(batch)
            self.batches = batches

            # Permanently lock unused nodes
            remaining_nodes = list(set([x for x in simulation.disease.graph.nodes]) - \
                              set(FLATTEN(self.batches)))
            for node in remaining_nodes:
                simulation.disease.graph.nodes[node]['data']['occupant']['locked'] = True

            self.current_batch = 0

        # Animal moved out due to infection
        for node in simulation.disease.graph.nodes:
            if simulation.disease.graph.nodes[node]['data']['occupant']['state'] == \
               simulation.disease.id_map['SY']:
                move_out(simulation, node)

        # Lock all empty rooms in non-active batches (even if they open up mid-run)
        for idx, batch in enumerate(self.batches):
            if idx != self.current_batch:
                for node in batch:
                    if simulation.disease.graph.nodes[node]['data']['occupant']['state'] == \
                       simulation.disease.id_map['E']:
                        simulation.disease.graph.nodes[node]['data']['occupant']['locked'] = True

        # Perform batch move operations
        if simulation.disease.time != 0 and simulation.disease.time % self.switch_time == 0:
            self.current_batch = (self.current_batch + 1) % self.number_of_batches

            # Move out oldest room and unlock
            for node in self.batches[self.current_batch]:
                move_out(simulation, node)
                simulation.disease.graph.nodes[node]['data']['occupant']['locked'] = False

        return self.swap_count

class SnakeIntervention(object):
    '''An intervention where a kennel ordering is determined and animals are
    placed in that kennel order like a queue. Once an animal reaches the end of
    the queue, they are removed from the simulation (placed in an independent area).
    If an animal becomes infected, they are removed to an independent area.

    Note: This intervention assumes that appropriate kennel order is the node order
    (i.e. 1, 2, 3, 4 is the queue order).
    '''

    def __init__(self, name='', leave_time=72):
        self.name = name
        self.swap_count = 0
        self.leave_time = leave_time

    def update(self, simulation): #pylint: disable=W0613
        '''Update the intervention given the simulation state.

        Arguments:
            simulation {Simulation} -- the current simulation state

        Returns:
            int -- the number of swaps updated
        '''
        get_node_state = lambda node: \
            simulation.disease.graph.nodes[node]['data']['occupant']['state']
        get_state_from_id = lambda id: \
            simulation.disease.id_map[id]
        node_ids = list(sorted([x for x in simulation.disease.graph.nodes], reverse=True))
        for node in node_ids: # Move out symptomatic nodes and timed nodes
            if get_node_state(node) == get_state_from_id('SY'):
                move_out(simulation, node)
            if simulation.disease.time - \
                    simulation.disease.graph.nodes[node]['data']['occupant']['intake_t'] > \
                    self.leave_time:
                move_out(simulation, node)

        found_gap = True
        while found_gap:
            found_gap = False
            for idx, node in enumerate(node_ids):
                if idx == 0:
                    continue
                elif get_node_state(node_ids[idx - 1]) == get_state_from_id('E') and \
                     get_node_state(node_ids[idx]) != get_state_from_id('E'):
                    simulation.disease.swap_cells(node, node_ids[idx - 1])
                    self.swap_count += 1
                    found_gap = True

        return self.swap_count

if __name__ == '__main__':
    from main import main
    main()
