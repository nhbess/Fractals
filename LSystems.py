import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image
from scipy.ndimage import convolve
from tqdm import tqdm
import Visuals

class Board:
    def __init__(self, n:int, m:int,l:int) -> None:
        self.n = n
        self.m = m
        self.l = l
        
        self.A = np.arange(0,2)
        self.O = np.random.choice(self.A)
        self.P = self._production_rules()
        self.W = np.random.randint(0, 2, len(self.P))        
        
        


        self.data = []
        
        self.B = np.zeros((n, m, l), dtype=int)

    def _production_rules(self) -> dict:
        # Agregar o quitar, esa es la question.
        N_RULES = 2 #number of production rules
        N_REACTANTS = 2
        N_PRODUCTS = 2
        P = []
        for _ in range(N_RULES):
            reactants = np.random.choice(self.A, N_REACTANTS)
            products = np.random.choice(self.A, N_PRODUCTS)
            P.append((reactants, products))
        return P

    def set_seeds(self, seeds:int) -> None:
        for _ in range(seeds):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.m)
            z = np.random.randint(0, self.l)
            self.B[x, y, z] = 1
    
    def update(self) -> None:

        kernel = np.random.randint(0, 2, (3, 3, 3))
        S = self.B.copy()
        N = convolve(S, kernel, mode='wrap')
        self.B[N == 1] = 1
        self.data.append(self.B.copy())

        if np.sum(self.W) == 0:
            print('DEAD')
        
        #chose P with probability W        
        active_production_rules = [p for p, w in zip(self.P, self.W) if w > 0]
        print(f'active_production_rules {active_production_rules}')
        sys.exit()
            

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000) 
    np.random.seed(seed)
    print(f'Seed: {seed}')
    ratio = 16/9
    Y = 50
    X = Y   #int(Y/ratio)
    Z = X
    RUNS = X

    for run in range(1):    
        b = Board(X, Y, Z)
        b.B[X//2, Y//2, Z//2] = 1 #set seed in the middle
        
        kernel = np.random.randint(0, 2, (3, 3, 3))
        for i in tqdm(range(RUNS)):
            b.update(kernel)

        data = b.data
        Visuals.create_visualization(data, filename=f'Test {seed} Run {run}', duration=100, title=f'Run {run}', gif=False, video=True, rotate=True)
