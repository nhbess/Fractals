import datetime
import os

import imageio
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm


class Board:
    def __init__(self, n:int, m:int) -> None:
        self.n = n
        self.m = m
        
        self.data = []
        
        self.B = np.zeros((n, m), dtype=int)
        self.data.append(self.B.copy())
    
    def update(self, kernels, reactants, products) -> None:
        for kernel, reactant, product in zip(kernels, reactants, products):        
            S = self.B.copy()
            N = convolve2d(S, kernel, mode='same', boundary='wrap')
            self.B[N == reactant] = product
            #print(f'N:\n{N}')
            #print(f'S:\n{S}')
            #print(f'B:\n{self.B}')
            self.data.append(self.B.copy())
            
if __name__ == '__main__':

    
    ratio = 16/9
    Y = 100
    X = int(Y/ratio)
    X = 100

    
    UPDATES = int(Y*1.5)
    UPDATES = 10
    
    print(f'X: {X}, Y: {Y}')

    for run in tqdm(range(1)):
        seed = np.random.randint(0, 100000000)
        
        np.random.seed(seed)
        print(f'Seed: {seed}')


        b = Board(X, Y)
        b.B[X//2, Y//2] = 1

        n_kernels = 10

        kernels = []
        for i in range(n_kernels):
            kernel = np.random.randint(0, 2, (i+3,i+3))
            kernels.append(kernel)
        #kernels = [np.random.randint(0, 2, (5,5)) for _ in range(n_kernels)]
        reactants = [np.random.randint(0, n_kernels) for _ in range(n_kernels)]
        products = [np.random.randint(0, n_kernels) for _ in range(n_kernels)]
        print(f'Kernel:\n{kernels}')


        #for i in tqdm(range(UPDATES)):
        for i in range(UPDATES):
            b.update(kernels, reactants, products)

        height, width, = b.data[0].shape
        images = []
        for img_data in b.data:
            img_data_scaled = (img_data * 255).astype(np.uint8)
            images.append(img_data_scaled)


        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = '_MEDIA_New'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        #imageio.mimsave(f'{folder_name}/{date}_{seed}_{run}.gif', images, duration=50)
        imageio.mimsave(f'test.gif', images, duration=5)