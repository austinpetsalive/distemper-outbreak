# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

class AggregatePlot(object):
    def __init__(self, disease, kennel):
        self.disease = disease
        self.kennel = kennel

        #QtGui.QApplication.setGraphicsSystem('raster')
        self.app = QtGui.QApplication([])
        #mw = QtGui.QMainWindow()
        #mw.resize(800,800)

        self.win = pg.GraphicsWindow(title="Aggregate Measures")
        self.win.resize(1000,600)
        self.win.setWindowTitle('Aggregate Measures')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        plots = []
        curves = []
        
        plots.append(self.win.addPlot(title="Population"))
        curves.append(plots[-1].plot(pen='b'))

        plots.append(self.win.addPlot(title="Infected"))
        curves.append(plots[-1].plot(pen='y'))

        self.win.nextRow()

        plots.append(self.win.addPlot(title="Survived/Immune"))
        curves.append(plots[-1].plot(pen='g'))

        plots.append(self.win.addPlot(title="Died"))
        curves.append(plots[-1].plot(pen='r'))

        self.curves = curves
        self.plots = plots
        self.data = {'pop': [0], 'inf': [0], 'imm': [0], 'die': [0]}
        #self.app.exec_()
        #QtGui.QApplication.instance().exec_()

    def update(self):
        self.app.processEvents()
        empty_nodes = len(self.disease.get_state_node('E')['members'])
        susceptible_nodes = len(self.disease.get_state_node('S')['members'])
        survived_nodes = len(self.disease.get_state_node('IS')['members'])
        infected_nodes = len(self.disease.get_state_node('I')['members'])
        died_nodes = len(self.disease.get_state_node('D')['members'])

        self.data['pop'].append(susceptible_nodes+survived_nodes+infected_nodes)
        self.data['inf'].append(infected_nodes)
        self.data['imm'].append(survived_nodes)
        self.data['die'].append(died_nodes)

        for curve, data in zip(self.curves, [self.data['pop'], self.data['inf'], self.data['imm'], self.data['die']]):
            curve.setData(data)

            
if __name__ == '__main__':
    from main import main
    main()