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

class Simulation(object):
    
    def __init__(self, params):
        self.FPS = 0
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 480
        self.params = params
    
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
        
        self.kennels = Kennels()
        self.disease = DistemperModel(self.kennels.get_graph(), self.params)

        self.update_hooks = []

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
        textpos.centerx = 100
        self.surface.blit(text, textpos)

    def update(self):
        self.check_events()
        self.surface.fill((255,255,255))

        if not self.disease.in_equilibrium():
            self.disease.update(self.kennels)
        self.kennels.draw(self.surface, self.disease)

        for hook in self.update_hooks:
            hook()

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

def main():
    params = {
        'pIntake': 0.25,
        'pInfect': 0.04,
        'pSurvive': 0.0025,
        'pDie': 0.0058333333333333,
        'pDieAlternate': 0.0,
        'refractoryPeriod': 3.0*24.0
    }
    sim = Simulation(params)
    plt = AggregatePlot(sim.disease, sim.kennels)
    sim.update_hooks.append(plt.update)
    sim.run()

if __name__ == '__main__':
    main()