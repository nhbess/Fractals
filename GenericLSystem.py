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
        self.data = [self.S]

        
    def _production_rules(self, n_production_rules) -> dict:
        N_REACTANTS = 1
        N_PRODUCTS = 4

        REACTANTS = []
        PRODUCTS = []

        CONTEXT_FREE = True
        VARIABLE_PRODUCTS = True

        for i in range(n_production_rules):
            if CONTEXT_FREE:
                if VARIABLE_PRODUCTS:
                    reactant = np.random.randint(0, len(self.A), np.random.randint(1, N_REACTANTS+1))
                    product = np.random.randint(0, len(self.A), np.random.randint(1, N_PRODUCTS+1))
                else:
                    reactant = np.random.randint(0, len(self.A), N_REACTANTS)
                    product = np.random.randint(0, len(self.A), N_PRODUCTS)
            else:
                if VARIABLE_PRODUCTS:
                    reactant = np.random.randint(0, len(self.A), np.random.randint(1, N_REACTANTS+1))
                    product = np.random.randint(0, len(self.A), np.random.randint(1, N_PRODUCTS+1))
                else:
                    reactant = np.random.randint(0, len(self.A), N_REACTANTS)
                    product = np.random.randint(0, len(self.A), N_PRODUCTS)

            if not any(np.array_equal(reactant, r) for r in REACTANTS):
                REACTANTS.append(reactant)
                PRODUCTS.append(product)
                print(f'Rule {len(REACTANTS)}: {reactant} -> {product}')

        P = [[r,p] for r,p in zip(REACTANTS, PRODUCTS)]
        return P
    
    def update(self) -> None:
        new_S = []
        for s in self.S:
            for rule in self.P:
                reactant, product = rule
                if np.all(reactant == s):
                    new_S.extend(product)
                    break  # Ensures only the first matching rule is applied
                    
        RESTRICT_SIZE = 100
        if len(new_S) > RESTRICT_SIZE:
            new_S = new_S[:RESTRICT_SIZE]
        self.S = np.array(new_S)
        self.data.append(self.S.copy())

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000) 
    #seed = 74340482
    np.random.seed(seed)
    print(f'Seed: {seed}')
    
    N_UPDATES = 100
    N_SYMBOLS = 10
    N_PRODUCTION_RULES = 10*2
    
    for run in range(1):
        b = GLS(n_symbols=N_SYMBOLS, n_production_rules=N_PRODUCTION_RULES)
        for i in range(N_UPDATES):
            b.update()


        data = b.data

        maximum_length = max([len(d) for d in data])
        if maximum_length < 4:
            continue
        n_updates = len(data)
        new_data = np.zeros((n_updates, maximum_length), dtype=int)
        for i, d in enumerate(data):
            new_data[i, :len(d)] = d

        #Visuals.plot_frame(new_data, f'_MEDIA_GLSYS/GLS_{seed}_{run}.png', cmap='Greens')
        Visuals.plot_frame(new_data, f'test.png', cmap='inferno')

    