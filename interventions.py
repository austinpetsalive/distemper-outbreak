"""This module contains the intervention objects.
"""

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
            if data['occupant'] is None:
                node_weights.append(1)
            elif data['occupant']['state'] == 4: # Infected and Symptomatic
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
