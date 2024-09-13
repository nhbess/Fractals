import numpy as np
from scipy.ndimage import convolve

import numpy as np

def find_pattern(grid, pattern):
    pattern_rows, pattern_cols = pattern.shape
    padded_grid = np.pad(grid, ((pattern_rows - 1, pattern_rows - 1), (pattern_cols - 1, pattern_cols - 1)), mode='wrap')
    
    
    grid_rows, grid_cols = grid.shape
    
    matches = []
    for i in range(1,grid_rows+1):
        for j in range(1, grid_cols+1):
            sub_grid = padded_grid[i:i + pattern_rows, j:j + pattern_cols] 
            if np.array_equal(sub_grid, pattern):
                matches.append((i - 1, j - 1))    
    return matches


# Example usage
grid = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

pattern = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

matches = find_pattern(grid, pattern)
print("Pattern found at positions:\n", matches)
