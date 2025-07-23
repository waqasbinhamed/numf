from scipy.signal import find_peaks
import numpy as np

def get_neighbors(locations: list[int], m: int, nrad: int = 2) -> list[int]:
    """
    Return a list of indices within nrad of each location, clipped to [0, m).

    Args:
        locations (list[int]): List of peak indices.
        m (int): Maximum index (exclusive).
        nrad (int): Neighborhood radius.

    Returns:
        list[int]: Unique indices within the specified neighborhood.
    """
    neighbors = set()
    for idx in locations:
        # Add indices in the neighborhood, step=1 for full coverage
        neighbors.update(range(max(0, idx - nrad), min(m, idx + nrad + 1)))
    return sorted(neighbors)

def get_peaks(M: np.ndarray, nrad: int = 2) -> list[int]:
    """
    Find likely peak neighborhoods in each column of M.

    Args:
        M (np.ndarray): 2D array (m, n).
        nrad (int): Neighborhood radius for peaks.

    Returns:
        list[int]: Sorted unique indices in the neighborhoods of detected peaks.
    """
    m, n = M.shape
    all_peaks = []
    for j in range(n):
        # Tune prominence and width parameters for your data
        peaks, _ = find_peaks(M[:, j], prominence=1, width=6)
        all_peaks.extend(peaks)
    return get_neighbors(all_peaks, m, nrad=nrad)
