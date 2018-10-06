import numpy as np

class SortIntervention(object):
    def __init__(self, name=''):
        self.name = name
        self.swap_count = 0

    def update(self, simulation):
        swap_list = self.compute_swap_map(simulation.disease.graph)
        for swap in swap_list:
            self.swap_count += 1
            simulation.disease.swap_cells(swap[0], swap[1])
        return self.swap_count

    def compute_swap_map(self, graph):
        swap_list = []
        node_weights = []
        for node in graph.nodes:
            data = graph.nodes[node]['data']
            if data['occupant'] is None:
                node_weights.append(1)
            elif data['occupant']['state'] == 4: # Infected and Symptomatic
                node_weights.append(2)
            elif data['occupant']['state'] == 2 or data['occupant']['state'] == 0 or data['occupant']['state'] == 5: # Not susceptible/empty/died
                node_weights.append(1)
            else:
                node_weights.append(data['occupant']['immunity'])
        weighted_nodes = list(zip(list(range(0, len(node_weights))), node_weights))

        # Node, this does NOT compute the most efficient swap path
        for j in range(0, len(weighted_nodes)):
            iMax = -1
            vMax = -1
            for i in range(j, len(weighted_nodes)):
                if weighted_nodes[i][1] > vMax:
                    vMax = weighted_nodes[i][1]
                    iMax = i
            
            if (iMax != j):
                tmp = weighted_nodes[j]
                weighted_nodes[j] = weighted_nodes[iMax]
                weighted_nodes[iMax] = tmp
                swap_list.append([j, iMax])
            
        return swap_list