import numpy as np
from scipy.signal import convolve2d
import sys

class LSys:
    def __init__(self, grid_size=(10, 10), N_SYMBOLS=5, N_RULES=3) -> None:        
        self.grid_size = grid_size
        self.n_symbols = N_SYMBOLS

        self.A = np.arange(0, N_SYMBOLS)                           # Alphabet (symbols)
        self.O = np.random.choice(self.A)                          # Axiom (initiator)
        self.P = self._make_rules(N_RULES)                     # Production rules
        
        self.B = np.zeros(grid_size, dtype=int)                    # Initialize the grid with zeros (empty grid)
        self.B[grid_size[0]//2, grid_size[1]//2] = 1          # Set the axiom in the middle of the grid
        self.data = [self.B.copy()]                                # Store the data for visualization
    
    def _make_rules(self, N_RULES) -> list:
        """Create random production rules."""
        P = []
        P.append([np.array([[0, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 0]]), 
                np.random.randint(0, self.n_symbols, (3, 3))])
        for _ in range(N_RULES):
            P.append([np.random.randint(0, self.n_symbols, (3, 3)), np.random.randint(0, self.n_symbols, (3, 3))])
        return P

    def update(self) -> None:
        S = self.B.copy()
        for reactant, product in self.P:
            #print(f'domain\n{reactant}')
            #print(f'image\n{product}')
            
            
            S = convolve2d(S, reactant, mode='same', boundary='wrap')
        
        
        self.B = S
        self.data.append(self.B.copy())


# Example usage

#set seeds
seed = np.random.randint(0, 1000)
np.random.seed(seed)

# good seeds = []
size = 500
lsystem = LSys(grid_size=(size, size), 
               N_SYMBOLS=2,
               N_RULES=1)

from tqdm import tqdm
for _ in tqdm(range(size*2)):
    lsystem.update()

print(seed)
#print(lsystem.data)
images = []

#set seeds in the board

for img_data in tqdm(lsystem.data):
    img_data_scaled = (img_data * 255).astype(np.uint8)    
    images.append(img_data_scaled)



# Save as a GIF
import imageio
imageio.mimsave(f'test.gif', images, duration=10)

print("Done, seed = ", seed)