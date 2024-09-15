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
np.set_printoptions(precision=2, suppress=True)

class GLS:
    def __init__(self, n_symbols:int =2, n_production_rules:int = 2, production_rules:np.array = None) -> None:

        self.A = np.arange(n_symbols)
        self.O = np.random.choice(self.A, 1)

        
        if production_rules is not None:    self.P = production_rules
        else:                               self.P = self._production_rules(n_production_rules)
        
        self.S = np.array(self.O)
        print(f'S: {self.S}')
        self.data = []

        
    def _production_rules(self, n_production_rules) -> dict:
        N_REACTANTS = 1
        N_PRODUCTS = 2

        REACTANTS = []
        PRODUCTS = []
        for i in range(n_production_rules):
            reactant = np.random.randint(0, len(self.A), N_REACTANTS)
            #product = np.random.randint(0, len(self.A), np.random.randint(1, N_PRODUCTS+1))
            product = np.random.randint(0, len(self.A), N_PRODUCTS)
            if not any(np.array_equal(reactant, r) for r in REACTANTS):
                REACTANTS.append(reactant)
                PRODUCTS.append(product)
        

        P = [[r,p] for r,p in zip(REACTANTS, PRODUCTS)]
        print(f'len P: {len(P)}')
        return P
    
    def update(self) -> None:
        new_S = []
        for s in self.S:
            for rule in self.P:
                reactant, product = rule
                if np.all(reactant == s):
                    new_S.extend(product)
                    break  # Ensures only the first matching rule is applied
                    
        self.S = np.array(new_S)
        self.data.append(self.S.copy())
        print(f'S: {self.S}')

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000) 

    np.random.seed(seed)
    print(f'Seed: {seed}')
    
    N_UPDATES = 5
    N_SYMBOLS = 2
    N_PRODUCTION_RULES = 5
    
    b = GLS(n_symbols=N_SYMBOLS, n_production_rules=N_PRODUCTION_RULES)
    for i in range(N_UPDATES):
        b.update()

    