
import numpy as np
def swap(coords: np.ndarray) -> np.ndarray:
   
    # Copy the original array to avoid modifying the original
    swapped_coords = np.copy(coords)
    # Swap x and y coordinates for each point
    swapped_coords[:, [0, 2]] = swapped_coords[:, [1, 3]]
    swapped_coords[:, [1, 3]] = swapped_coords[:, [0, 2]]
    return swapped_coords
