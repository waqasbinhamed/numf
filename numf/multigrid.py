import math
import numpy as np

def create_restriction_matrix(m: int) -> np.ndarray:
    """
    Create a restriction (downsampling) matrix for multigrid methods.

    Args:
        m (int): Size of the fine grid.

    Returns:
        np.ndarray: Restriction matrix of shape (floor(m/2), m).
    """
    n = int(math.floor(m / 2))
    R = np.zeros((n, m))
    for i in range(n):
        # Average over three consecutive fine grid points
        R[i, 2 * i: 2 * i + 3] = 1 / 3
    return R

def get_fine_indices(indices: list[int], scaling_factor: int = 2) -> np.ndarray:
    """
    Map coarse grid indices to fine grid indices by scaling.

    Args:
        indices (list[int]): Indices on the coarse grid.
        scaling_factor (int): Factor to scale indices.

    Returns:
        np.ndarray: Corresponding indices on the fine grid.
    """
    return np.array(indices) * scaling_factor