"""This module contains the aggregate visualization of the simulation.
"""

import sys

import pyqtgraph as pg # pylint: disable=E0401
from pyqtgraph.Qt import QtCore, QtGui # pylint: disable=E0401


class AggregatePlot(object):
    '''This class renders aggregate variables from a disease simulation.
    '''

    def __init__(self, disease, kennel):
        self.disease = disease
        self.kennel = kennel

        self.app = QtGui.QApplication([])

        self.win = pg.GraphicsWindow(title="Aggregate Measures")
        self.win.resize(1500, 600)
        self.win.setWindowTitle('Aggregate Measures')
        self.win.keyPressEvent = self.keyPressEvent

        pg.setConfigOptions(antialias=True)

        plots = []
        curves = []

        plots.append(self.win.addPlot(title="Total Population"))
        curves.append(plots[-1].plot(pen='b'))

        plots.append(self.win.addPlot(title="Current Population"))
        curves.append(plots[-1].plot(pen='b'))

        plots.append(self.win.addPlot(title="Total Infected"))
        curves.append(plots[-1].plot(pen='y'))

        plots.append(self.win.addPlot(title="Current Infected"))
        curves.append(plots[-1].plot(pen='y'))

        plots.append(self.win.addPlot(title="Infection Rate"))
        curves.append(plots[-1].plot(pen=(255, 165, 0)))

        self.win.nextRow()

        plots.append(self.win.addPlot(title="Total Survived/Immune"))
        curves.append(plots[-1].plot(pen='g'))

        plots.append(self.win.addPlot(title="Current Survived/Immune"))
        curves.append(plots[-1].plot(pen='g'))

        plots.append(self.win.addPlot(title="Total Died"))
        curves.append(plots[-1].plot(pen='r'))

        plots.append(self.win.addPlot(title="Current Died"))
        curves.append(plots[-1].plot(pen='r'))

        plots.append(self.win.addPlot(title="Survival Rate"))
        curves.append(plots[-1].plot(pen='w'))

        self.curves = curves
        self.plots = plots
        self.data = {'pop': [0], 'inf': [0], 'imm': [0], 'die': [0],
                     'tpop': [0], 'tinf': [0], 'timm': [0], 'tdie': [0],
                     'ir': [0], 'sr': [0]}

    def _key_press_event(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            sys.exit(0)

    keyPressEvent = _key_press_event

    def update(self):
        '''Update the graph.
        '''

        self.app.processEvents()

        self.data['tpop'].append(self.disease.total_intake)
        self.data['tinf'].append(self.disease.total_infected)
        self.data['timm'].append(self.disease.total_discharged)
        self.data['tdie'].append(self.disease.total_died)

        susceptible_nodes = len(self.disease.get_state_node('S')['members'])
        survived_nodes = len(self.disease.get_state_node('IS')['members'])
        infected_nodes = len(self.disease.get_state_node('I')['members'])
        symptomatic_nodes = len(self.disease.get_state_node('SY')['members'])
        died_nodes = len(self.disease.get_state_node('D')['members'])

        self.data['pop'].append(susceptible_nodes +
                                survived_nodes +
                                infected_nodes +
                                symptomatic_nodes +
                                died_nodes)
        self.data['inf'].append(infected_nodes + symptomatic_nodes)
        self.data['imm'].append(survived_nodes)
        self.data['die'].append(died_nodes)

        self.data['ir'].append(self.disease.total_infected/self.disease.total_intake)
        self.data['sr'].append(self.disease.total_discharged/self.disease.total_intake)

        for curve, data in zip(self.curves, [self.data['tpop'], self.data['pop'],
                                             self.data['tinf'], self.data['inf'], self.data['ir'],
                                             self.data['timm'], self.data['imm'],
                                             self.data['tdie'], self.data['die'], self.data['sr']]):
            curve.setData(data)


if __name__ == '__main__':
    from main import main
    main()
