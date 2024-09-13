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
        
        self.data = []
        
        self.B = np.zeros((n, m, l), dtype=int)

    def set_seeds(self, seeds:int) -> None:
        for _ in range(seeds):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.m)
            z = np.random.randint(0, self.l)
            self.B[x, y, z] = 1
    
    def update(self, kernel=None) -> None:
        if kernel is None:
            kernel = np.random.randint(0, 2, (3, 3, 3))
            
        S = self.B.copy()
        N = convolve(S, kernel, mode='wrap')
        self.B[N == 1] = 1
        self.data.append(self.B.copy())

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
        Visuals.create_visualization_pyvista(data, filename=f'Test {seed} Run {run}', duration=100, title=f'Run {run}', gif=False, video=True, rotate=True)