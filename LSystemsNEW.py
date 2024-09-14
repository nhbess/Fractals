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

class LS:
    def __init__(self, n:int, m:int, n_production_rules:int = 2, production_rules = None) -> None:
        if production_rules is not None:    self.P = production_rules
        else:                               self.P = self._production_rules(n_production_rules)
        
        self.B = np.zeros((n,m), dtype=int)
        self.B[n//2, m//2] = 1
        self.data = []


    def _find_pattern(self, grid, pattern):
        pattern_rows, pattern_cols = pattern.shape
        padded_grid = np.pad(grid, ((pattern_rows - 1, pattern_rows - 1), (pattern_cols - 1, pattern_cols - 1)), mode='wrap')
        
        grid_rows, grid_cols = grid.shape
        
        M = np.zeros((grid_rows, grid_cols), dtype=int)
        for i in range(1,grid_rows+1):
            for j in range(1, grid_cols+1):
                sub_grid = padded_grid[i:i + pattern_rows, j:j + pattern_cols] 
                if np.array_equal(sub_grid, pattern):
                    M[i - 1, j - 1] = 1
        
        return M

    def _production_rules(self, n_production_rules) -> dict:
        P = []
        P.append([  np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [0, 0, 0]]), 
                    np.random.randint(0, 2, (3, 3))]) #First rule

        for _ in range(n_production_rules):
            reactants = np.random.randint(0, 2, (3, 3))
            products = np.random.randint(0, 2, (3, 3))
            P.append([reactants, products])

        return P
    
    def update(self) -> None:
        S = self.B.copy()
        for rule in self.P:
            reactant, product = rule
            reactant = reactant.clip(0, 1)
            reactant = np.where(reactant > 0, 0, reactant)
            M = self._find_pattern(S, reactant)
            if sum(sum(M)) == 0:
                continue
            #print(f'pattern\n{M}')
            
            N = convolve(M, product, mode='wrap')
            self.B[N == 1] = 1
            # where N == 1, set to 1 in B where N == 0, leave deduct 1 in B
            #self.B[N == 0] = np.where(self.B[N == 0] == 0, 0, self.B[N == 0] - 1)
            #self.B = np.where(N == 1, 1, self.B)
            #self.B = np.where(N == 0, np.where(self.B == 0, 0, self.B - 1), self.B)
            #self.B = np.clip(self.B, 0, 1)

        self.data.append(self.B.copy())

            

if __name__ == '__main__':
    pass
    #seed = np.random.randint(0, 100000000) 
    #
    #np.random.seed(seed)
    #print(f'Seed: {seed}')
    #Y = 50
    #X = Y   #int(Y/ratio)
    #RUNS = X
    #N_PRODUCTION_RULES = 100
    #for run in range(1):    
    #    b = Board(X, Y, N_PRODUCTION_RULES)
    #    for i in range(Y*2):
    #        b.update()
#
    #    data = b.data
    #    Visuals.create_visualization_grid(data, filename=f'Test', duration=100, gif=True, video=False)