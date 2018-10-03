import pygame
import sys
import time
import random

import pygame.locals as pgl

import numpy as np

import networkx as nx
import epydemic

from custom_disease_model import Kennels, DistemperModel
from aggregate_visualization import AggregatePlot

from copy import copy

from threading import Thread

import logging

import multiprocessing
from multiprocessing import Pool

import tqdm

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

class Simulation(object):
    
    def __init__(self, params, spatial_visualization=True, aggregate_visualization=True, return_on_equillibrium=False):
        self.return_on_equillibrium = return_on_equillibrium
        self.spatial_visualization = spatial_visualization
        self.aggregate_visualization = aggregate_visualization

        if not self.spatial_visualization and not self.aggregate_visualization and not self.return_on_equillibrium:
            logging.warning('Warning: No visualizations were set, it is highly recommended you set return_on_equillibrium to True otherwise you will have to manually manage the simulation state.')

        self.params = params
    
        self.kennels = Kennels()
        self.disease = DistemperModel(self.kennels.get_graph(), self.params)

        self.update_hooks = []

        if spatial_visualization:
            self.FPS = 0
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 480
            pygame.init()
            self.fpsClock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
            self.surface = pygame.Surface(self.screen.get_size())
            self.surface = self.surface.convert()
            self.surface.fill((255,255,255))
            self.clock = pygame.time.Clock()

            pygame.key.set_repeat(1, 40)
        
            self.screen.blit(self.surface, (0,0))

            self.font = pygame.font.Font(None, 36)
        
        if aggregate_visualization:
            self.plt = AggregatePlot(self.disease, self.kennels)
            self.update_hooks.append(self.plt.update)

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pgl.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pgl.KEYDOWN:
                if event.key == pgl.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    def redraw(self):
        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()
        pygame.display.update()
        self.fpsClock.tick(self.FPS)

    def draw_ui(self):
        text = self.font.render('{0} days, {1} hours'.format(int(np.floor(self.disease.t/24.0)), self.disease.t%24), 1, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = 200
        self.surface.blit(text, textpos)

    def get_disease_state(self):
        return {sc: len(self.disease.get_state_node(sc)['members']) for sc in self.disease.id_map.keys()}
        
    def update(self):
        if self.spatial_visualization:
            self.check_events()
            self.surface.fill((255,255,255))

        if not self.disease.in_equilibrium():
            self.disease.update(self.kennels)
        elif self.return_on_equillibrium:
            self.running = False
            return

        for hook in self.update_hooks:
            hook()
        
        if self.spatial_visualization:
            self.kennels.draw(self.surface, self.disease)
            self.draw_ui()
            self.redraw()
    
    def stop(self):
        self.running = False

    def run(self, asynchronous=False):
        self.running = True
        if asynchronous:
            self.async_thread = Thread(target=self.run, args=(False,))
            self.async_thread.start()
        else:
            while self.running:
                self.update()
            return self.get_disease_state()

class BatchSimulation(object):
    def __init__(self, params, runs, pool_size=-1):
        self.params = params
        self.runs = runs
        if pool_size == None:
            self.pool_size = 1
        elif pool_size <= 0:
            self.pool_size = multiprocessing.cpu_count()
        
    def run(self):
        results = []
        with Pool(self.pool_size) as p:
            for i in tqdm.tqdm(p.imap_unordered(BatchSimulation.run_simulation, [self.params]*self.runs), total=self.runs):
                results.append(i)
            p.close()
            p.join()
        return results
            
    @staticmethod
    def run_simulation(params):
        return Simulation(params, spatial_visualization=False, aggregate_visualization=False, return_on_equillibrium=True).run()

def main(batch=False):
    params = {
            'pIntake': 0.25,
            'pInfect': 0.04,
            'pSurvive': 0.0025,
            'pDie': 0.0058333333333333,
            'pDieAlternate': 0.0,
            'refractoryPeriod': 3.0*24.0
        }
    if not batch:
        sim = Simulation(params, spatial_visualization=False, return_on_equillibrium=True, aggregate_visualization=False)
        print(sim.run())
    else:
        runs = 32
        bar_width = 0.35
        proportion = True
        colors = [cm.jet(0), cm.jet(0.5)]
        alphas = [0.5, 0.25]
        labels = ['Rapid Intake', 'Slow Intake']
        
        params1 = copy(params)
        params1['pIntake'] = 0.1
        
        results = BatchSimulation(params, runs).run()

        total = sum(list(results[0].values()))
        df = pd.DataFrame.from_records(results)
        if proportion:
            df /= total
        
        plt.rcdefaults()

        objects = df.columns
        y_pos = np.arange(len(objects))
        
        plt.bar(y_pos-bar_width/2, df.mean(), bar_width, align='center', alpha=alphas[0], yerr=df.std()/np.sqrt(len(df)), color=colors[0], label=labels[0])
        
        results = BatchSimulation(params1, runs).run()
        df = pd.DataFrame.from_records(results)
        if proportion:
            df /= total
        plt.bar(y_pos+bar_width/2, df.mean(), bar_width, align='center', alpha=alphas[1], yerr=df.std()/np.sqrt(len(df)), color=colors[1], label=labels[1])

        plt.xticks(y_pos, objects)
        plt.ylabel('Mean Animal Count')
        plt.title('Average Simulation Performance')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main(batch=True)