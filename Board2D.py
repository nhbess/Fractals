import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from PIL import Image
from scipy.signal import convolve2d
from tqdm import tqdm

class Board:
    def __init__(self, n:int, m:int) -> None:
        self.n = n
        self.m = m
        
        self.data = []
        
        self.B = np.zeros((n, m), dtype=int)
        self.data.append(self.B.copy())

    def set_seeds(self, seeds:int) -> None:
        for _ in range(seeds):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.m)
            self.B[x, y] = 1
    
    def update(self, kernel=None) -> None:
        if kernel is None:
            kernel = np.array([[0, 1, 0],
                               [0, 0, 0],
                               [1, 0, 1]])
        
        S = self.B.copy()
        N = convolve2d(S, kernel, mode='same', boundary='wrap')
        self.B[N == 1] = 1
        self.data.append(self.B.copy())



for run in tqdm(range(1)):
    seed = np.random.randint(0, 100000000)
    seed = 79945574
    np.random.seed(seed)
    print(f'Seed: {seed}')

    ratio = 16/9
    Y = 1000
    X = int(Y/ratio)

    print(f'X: {X}, Y: {Y}')

    b = Board(X, Y)
    #b.set_seeds(1)
    b.B[X//2, Y//2] = 1
    kernel = np.random.randint(0, 2, (3, 3))
    print(f'Kernel:\n{kernel}')
    

    for i in tqdm(range(int(Y*1.5))):
        b.update(kernel)

    height, width, = b.data[0].shape
    images = []
    for img_data in b.data:
        img_data_scaled = (img_data * 255).astype(np.uint8)
        images.append(img_data_scaled)
        
    # Save as a GIF
    imageio.mimsave(f'ratio_2.gif', images, duration=50)
    
