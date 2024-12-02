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
    
    def update(self, kernel=None) -> None:
        if kernel is None:
            kernel = np.array([[0, 1, 0],
                               [0, 0, 0],
                               [1, 0, 1]])
        
        S = self.B.copy()
        N = convolve2d(S, kernel, mode='same', boundary='wrap')
        self.B[N == 1] = 1
        self.data.append(self.B.copy())

if __name__ == '__main__':

    for run in tqdm(range(1)):
        seed = np.random.randint(0, 100000000)
        np.random.seed(seed)
        print(f'Seed: {seed}')

        ratio = 16/9
        Y = 500
        X = int(Y/ratio)

        print(f'X: {X}, Y: {Y}')

        b = Board(X, Y)
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


        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = '_MEDIA_New'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        imageio.mimsave(f'{folder_name}/{date}_{seed}_{run}.gif', images, duration=50)