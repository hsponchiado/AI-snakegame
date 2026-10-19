import random 
import numpy
import torch 
import pygame 
from sys import exit

pygame.init()
window = pygame.display.set_mode((500,500))

while True: 
    for event in pygame.event.get():
        if event.type == pygame.quit():
            pygame.quit()
            exit()





