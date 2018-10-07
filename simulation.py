"""This is the module for the primary simulation objects. Simulation is
for single simulations (with visualization) and BatchSimulation is for
experiments with multiple runs.
"""

import logging
import multiprocessing
from multiprocessing.pool import Pool
import sys
from copy import deepcopy
from threading import Thread

import numpy as np
import pygame
import pygame.locals as pgl
import tqdm

from aggregate_visualization import AggregatePlot
from custom_disease_model import DistemperModel, Kennels


class Simulation(object):
    '''This is the primary simulation class.
    It is responsible for both computation and rendering.
    '''

    def __init__(self, params,
                 spatial_visualization=True,
                 aggregate_visualization=True,
                 return_on_equillibrium=False):
        self.return_on_equillibrium = return_on_equillibrium
        self.spatial_visualization = spatial_visualization
        self.aggregate_visualization = aggregate_visualization

        if not self.spatial_visualization and \
           not self.aggregate_visualization and \
           not self.return_on_equillibrium:
            #pylint: disable=W1201
            logging.warning(('Warning: No visualizations were set, it is ' +
                             'highly recommended you set return_on_equillibrium ' +
                             'to True otherwise you will have to manually manage ' +
                             'the simulation state.'))

        self.params = params
        if 'infection_kernel_function' in self.params and \
           isinstance(self.params['infection_kernel_function'], str):
            self.params['infection_kernel_function'] = \
            eval(self.params['infection_kernel_function']) #pylint: disable=W0123
        else:
            self.params['infection_kernel_function'] = lambda node, k: 0.0
        if 'intervention' in self.params and isinstance(self.params['intervention'], str):
            self.params['intervention'] = \
            eval(self.params['intervention']) #pylint: disable=W0123
        else:
            self.params['intervention'] = None
        self.kennels = Kennels()
        self.disease = DistemperModel(self.kennels.get_graph(), self.params)

        self.update_hooks = []

        if spatial_visualization:
            self.fps = 0
            self.screen_width, self.screen_height = 640, 480
            pygame.init()
            self.fps_clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            self.surface = pygame.Surface(self.screen.get_size())
            self.surface = self.surface.convert()
            self.surface.fill((255, 255, 255))
            self.clock = pygame.time.Clock()

            pygame.key.set_repeat(1, 40)

            self.screen.blit(self.surface, (0, 0))

            self.font = pygame.font.Font(None, 36)

            self.running = False
            self.async_thread = None

        if aggregate_visualization:
            self.plt = AggregatePlot(self.disease, self.kennels)
            self.update_hooks.append(self.plt.update)

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pgl.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pgl.KEYDOWN:
                if event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

    def _redraw(self):
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        self.fps_clock.tick(self.fps)

    def _draw_ui(self):
        text = self.font.render('{0} days, {1} hours'.format(int(np.floor(self.disease.time/24.0)),
                                                             self.disease.time%24), 1, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = 200
        self.surface.blit(text, textpos)

    def _get_disease_state(self):
        return {sc: len(self.disease.get_state_node(sc)['members']) for sc in self.disease.id_map}

    def update(self):
        '''Update the simulation and redraw.
        '''

        if self.spatial_visualization:
            self._check_events()
            self.surface.fill(self.kennels.background_color)

        if not self.disease.in_equilibrium() and not self.disease.end_conditions():
            if 'intervention' in self.params and self.params['intervention'] is not None:
                self.params['intervention'].update(simulation=self)
            self.disease.update()
            for hook in self.update_hooks:
                hook()
        elif self.return_on_equillibrium:
            self.running = False
            return

        if self.spatial_visualization:
            self.kennels.draw(self.surface, self.disease)
            self._draw_ui()
            self._redraw()

    def stop(self):
        '''Stop the simulation.
        '''

        self.running = False

    def run(self, asynchronous=False):
        '''Run the simulation (async creates a new thread).

        Keyword Arguments:
            asynchronous {bool} -- if True, a new thread is created (default: {False})

        Returns:
            list(int) -- a list of the final counts of the different states
        '''

        self.running = True
        if asynchronous:
            self.async_thread = Thread(target=self.run, args=(False,))
            self.async_thread.start()
        else:
            while self.running:
                self.update()
            return self._get_disease_state()

class BatchSimulation(object):
    '''This class runs a batch version of the simulation.
    '''

    def __init__(self, params, runs, pool_size=-1):
        self.params = params
        self.runs = runs
        if pool_size is None:
            self.pool_size = 1
        elif pool_size <= 0:
            self.pool_size = multiprocessing.cpu_count()

    def run(self):
        '''This function runs the simulation asynchronously in multiple threads.

        Returns:
            list(list(int)) -- a list of all the results from the simulations
        '''

        results = []
        with Pool(self.pool_size) as thread_pool:
            tasks = tqdm.tqdm(thread_pool.imap_unordered(BatchSimulation.run_simulation,
                                                         [deepcopy(self.params) for _ in
                                                          range(0, self.runs)]),
                              total=self.runs)
            for i in tasks:
                results.append(i)
            thread_pool.close()
            thread_pool.join()
        return results

    @staticmethod
    def run_simulation(params):
        '''Run the simulation with a set of parameters

        Arguments:
            params {dict} -- the parameter dictionary for the simulation

        Returns:
            list(int) -- the simulation results
        '''

        return Simulation(params,
                          spatial_visualization=False,
                          aggregate_visualization=False,
                          return_on_equillibrium=True).run()
