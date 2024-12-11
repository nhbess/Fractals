import datetime
import os
import sys
import imageio
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
from Util import load_simple_image_as_numpy_array

class Board:
    def __init__(self, n:int, m:int) -> None:
        self.n = n
        self.m = m
        
        self.data = []
        
        self.B = np.zeros((n, m), dtype=int)
        self.data.append(self.B.copy())
    
    def update_L(self, kernels, reactants, products) -> None:
        for kernel, reactant, product in zip(kernels, reactants, products):        
            S = self.B.copy()
            N = convolve2d(S, kernel, mode='same', boundary='wrap')
            self.B[N == reactant] = product
            #print(f'N:\n{N}')
            #print(f'S:\n{S}')
            #print(f'B:\n{self.B}')
            self.data.append(self.B.copy())
    
    def update_simple(self, kernel) -> None:    
        S = self.B.copy()
        N = convolve2d(S, kernel, mode='same', boundary='wrap')
        #self.B[N == 1] = 1
        self.B = self.B + N
        self.B = np.clip(self.B, -1, 1)
        self.data.append(self.B.copy())

    def update_multikernel(self, kernels) -> None:
        for kernel in kernels:
            self.update_simple(kernel)


def make_gif(b: Board) -> None:
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

def make_gif_color(b: Board) -> None:
    height, width = b.data[0].shape
    images = []
    for img_data in b.data:
        # Ensure img_data is in float to prevent overflow
        img_data = img_data.astype(np.float64)
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val > min_val:
            # Normalize when there's a range
            img_data_normalized = (img_data - min_val) / (max_val - min_val)
        else:
            # Handle edge case where all values are the same
            img_data_normalized = np.zeros_like(img_data)

        # Convert normalized data to 8-bit format for visualization
        img_data_scaled = (img_data_normalized * 255).astype(np.uint8)
        images.append(img_data_scaled)
    
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = '_MEDIA_New'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Save the GIF
    #imageio.mimsave(f'{folder_name}/{date}.gif', images, duration=5)
    imageio.mimsave(f'test.gif', images, duration=5)



def expand_array(input_array):
    """
    Expands an NxM array into a (2N)x(2M) array by replicating each element in a 2x2 block.
    
    Parameters:
        input_array (np.ndarray): The input array of size NxM.
    
    Returns:
        np.ndarray: The expanded array of size (2N)x(2M).
    """
    input_array = np.asarray(input_array)  # Ensure input is a NumPy array
    return np.repeat(np.repeat(input_array, 2, axis=0), 2, axis=1)



if __name__ == '__main__':
    seed = np.random.randint(0, 100000000)
    seed = 41217378
    
    np.random.seed(seed)
    print(f'Seed: {seed}')

    mario = load_simple_image_as_numpy_array('assets/Mario.png')    

    ratio = 16/9
    Y = 100
    X = int(Y/ratio)
    X = 100

    UPDATES = int(Y*1.5)
    
    print(f'X: {X}, Y: {Y}')
    
    
    
    b = Board(X, Y)
    b.B[X//2, Y//2] = 1
    #b.B = np.random.randint(-1,2,(X, Y))

    N_KERNELS = 20
    kernels = []
    for i in range(N_KERNELS):
        kernel = np.random.random((3, 3))*2 - 1
        #kernels.append(kernel)
        #print(f'Kernel {i}:\n{kernel}')
        #kernels.append(kernel.T)
        #kernels.append(kernel.T.T)
        #kernels.append(kernel.T.T.T)
        kernel_expanded = expand_array(kernel)
        for j in range(1,1):
            print(f'Kernel expanded{i}:\n{kernel_expanded.shape}')
            #kernels.append(kernel_expanded)
            kernel_expanded = expand_array(kernel_expanded)
        
        kernels.append(kernel_expanded)
        kernels.append(kernel_expanded.T)
        kernels.append(kernel_expanded.T.T)
        kernels.append(kernel_expanded.T.T.T)
        
    kernels[-1] = np.random.random((3, 3))*2 - 1
    for i in tqdm(range(UPDATES)):
        b.update_multikernel(kernels)
        

    make_gif_color(b)