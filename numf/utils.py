import numpy as np

def create_unimodal_matrix(m: int, p: int) -> np.ndarray:
    """
    Create a unimodal restriction matrix of size (m, m).

    Args:
        m (int): Total size of the matrix.
        p (int): Peak index for unimodality.

    Returns:
        np.ndarray: Unimodal restriction matrix.
    """
    D = np.diag(np.ones(p + 1)) + np.diag(-1 * np.ones(p), -1)
    if p < m - 1:
        Dt = np.diag(np.ones(m - p - 1)) + np.diag(-1 * np.ones(m - p - 2), 1)
        Up = np.block([
            [D, np.zeros((p + 1, m - p - 1))],
            [np.zeros((m - p - 1, p + 1)), Dt]
        ])
    else:
        Up = D
    return Up

def create_difference_matrix(m: int) -> np.ndarray:
    """
    Create a first order difference matrix of size (m - 1, m).

    Args:
        m (int): Number of rows/columns.

    Returns:
        np.ndarray: First order difference matrix.
    """
    D = np.eye(m, k=1) - np.eye(m)
    return D[:-1]

def initialize_matrices(m: int, n: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize nonnegative matrices W (m, r) and H (r, n) with random values.

    Args:
        m (int): Number of rows in W.
        n (int): Number of columns in H.
        r (int): Rank of the factorization.

    Returns:
        tuple[np.ndarray, np.ndarray]: Initialized matrices W and H.
    """
    return np.random.rand(m, r), np.random.rand(r, n)
