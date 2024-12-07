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
        self.B[N == 1] = 1
        self.data.append(self.B.copy())

    def update_multikernel(self, kernels) -> None:
        for kernel in kernels:
            self.update_simple(kernel)




if __name__ == '__main__':
    import numpy as np

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


    mario = load_simple_image_as_numpy_array('assets/Mario.png')    
    N_KERNELS = 2
    kernels = [mario]
    
    for i in range(1,N_KERNELS+1):
        kernel = expand_array(kernels[-1])
        kernels.append(kernel)
        print(f'Kernel {i} {kernel.shape}:\n{kernel}')


    reactants = [int(np.sum(kernel)) for kernel in kernels]
    products = [k for k in kernels]
    products[-1] = [0]    
    print(f'Reactants: {reactants}')
    print(f'Products: {products}')
    #kernels = kernels[::-1]
    #for i in range(1,N_KERNELS+1):
    #    kernel = expand_array(kernels[-1])
    #    kernels.append(kernel)
    #    print(f'Kernel {i} {kernel.shape}:\n{kernel}')

    ratio = 16/9
    Y = 200
    X = int(Y/ratio)
    X = 200

    
    UPDATES = int(Y*1.5)
    UPDATES = 10
    
    print(f'X: {X}, Y: {Y}')

    for run in [1]:
        seed = np.random.randint(0, 100000000)
        
        np.random.seed(seed)
        print(f'Seed: {seed}')


        b = Board(X, Y)
        b.B[X//2, Y//2] = 1


        for i in tqdm(range(UPDATES)):
            #b.update_multikernel(kernels)
            b.update_L(kernels, reactants, products)
            
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