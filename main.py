#!/usr/bin/env python

import pygame
import sys
import time
import random

from pygame.locals import *

import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]

FPS = 5
pygame.init()
fpsClock=pygame.time.Clock()

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
surface = pygame.Surface(screen.get_size())
surface = surface.convert()
surface.fill((255,255,255))
clock = pygame.time.Clock()

pygame.key.set_repeat(1, 40)
  
screen.blit(surface, (0,0))

def draw_box(surf, color, pos, size):
    r = pygame.Rect((pos[0], pos[1]), (size[0], size[1]))
    pygame.draw.rect(surf, color, r)

def draw_line(surf, color, pos0, pos1, width=1):
    pygame.draw.line(surf, color, pos0, pos1)

def get_nodes_center(node, size):
    return (node['x']+size/2.0, node['y']+size/2.0)

class Kennels(object):
    def __init__(self):
        self.grid = [[12, 12], [12, 12], [0], [12, 12], [12, 12]]
        self.num_kennels = np.sum(flatten(self.grid))
        self.node_size = 10
        self.node_minor_pad = 1
        self.node_major_pad = 5
        self.x_offset = 50
        self.y_offset = 50
        self.nodes = []
        self.edges = []
        count = 0
        row_offset = 0
        for row in self.grid:
            col_offset = 0
            for segment_length in row:
                for i in range(0, segment_length):
                    new_node = {'node_id': count, 'x': col_offset + self.x_offset, 'y': row_offset + self.y_offset, 'color': (0, 0, 0), 'value': 0}
                    new_node['center'] = get_nodes_center(new_node, self.node_size)
                    self.nodes.append(new_node)
                    if i != segment_length - 1:
                        self.edges.append({'start': count, 'end': count + 1})
                    count += 1
                    col_offset += self.node_minor_pad + self.node_size
                col_offset += self.node_major_pad
            row_offset += self.node_major_pad + self.node_size

    def draw(self, surf):
        for node in self.nodes:
            draw_box(surf, node['color'], (node['x'], node['y']), [self.node_size, self.node_size])
        for edge in self.edges:
            draw_line(surf, (255, 0, 0), self.nodes[edge['start']]['center'], self.nodes[edge['end']]['center'])

if __name__ == '__main__':
    kennels = Kennels()

    while True:

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    pass
                elif event.key == K_DOWN:
                    pass
                elif event.key == K_LEFT:
                    pass
                elif event.key == K_RIGHT:
                    pass


        surface.fill((255,255,255))
        
        kennels.draw(surface)

        font = pygame.font.Font(None, 36)
        text = font.render(str(kennels.num_kennels), 1, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = 20
        surface.blit(text, textpos)
        screen.blit(surface, (0,0))

        pygame.display.flip()
        pygame.display.update()
        fpsClock.tick(FPS)