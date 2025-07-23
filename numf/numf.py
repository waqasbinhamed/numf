import numpy as np
from numf.utils import create_difference_matrix, create_unimodal_matrix

np.random.seed(42)

def numf(
    M: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    pvals: list[int] = None,
    l2: float = 0,
    beta: float = 0,
    iters: int = 100,
    save_file: str = None,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Run the NuMF algorithm to factorize M into W and H with unimodal constraints.

    Args:
        M (np.ndarray): Input data matrix (m, n).
        W (np.ndarray): Initial W matrix (m, r).
        H (np.ndarray): Initial H matrix (r, n).
        pvals (list[int], optional): List of peak indices to try for unimodality.
        l2 (float): L2 regularization parameter.
        beta (float): Additional regularization parameter.
        iters (int): Number of iterations.
        save_file (str, optional): Path to save W, H, and pouts.
        verbose (bool): Print progress if True.

    Returns:
        tuple: (W, H, pouts) after optimization.
    """
    m, n = M.shape
    r = W.shape[1]

    for it in range(1, iters + 1):
        pouts = numf_it(H, M, W, l2, m, n, pvals, r, beta)
        if it % 5 == 0 or it == iters:
            if verbose:
                loss = np.linalg.norm(M - W @ H, 'fro') / np.linalg.norm(M, 'fro')
                print(f"Loss: {loss}")
            if save_file is not None:
                with open(save_file, 'wb') as fout:
                    np.savez_compressed(fout, W=W, H=H, pouts=pouts)
                if verbose:
                    print(f'W and H matrices saved in {save_file} after {it} iterations.')
    return W, H, pouts

def numf_it(
    H: np.ndarray,
    M: np.ndarray,
    W: np.ndarray,
    l2: float,
    m: int,
    n: int,
    pvals: list[int],
    r: int,
    beta: float
) -> list[int]:
    """
    One iteration of block coordinate descent for NuMF.

    Returns:
        list[int]: List of selected peak indices for each component.
    """
    pouts = []
    Mi = M - W @ H
    for i in range(r):
        wi = W[:, i].reshape(m, 1)
        hi = H[i, :].reshape(1, n)

        Mi = Mi + wi @ hi  # Add back current component

        # Update hi (row of H)
        H[i, :] = update_hi(Mi, wi, n)

        # Update wi (column of W) and get best peak index
        W[:, i], pout = update_wi(Mi, wi, hi, m, pvals, l2, beta)
        pouts.append(pout)

        Mi = Mi - wi @ hi  # Subtract updated component
    return pouts

def update_wi(
    Mi: np.ndarray,
    wi: np.ndarray,
    hi: np.ndarray,
    m: int,
    pvals: list[int] = None,
    l2: float = 0,
    beta: float = 0
) -> tuple[np.ndarray, int]:
    """
    Update the value of w(i) column as part of BCD.

    Returns:
        tuple: (updated wi, best peak index)
    """
    wmin = np.empty((m, 1))
    min_score = np.Inf
    min_p = 0

    hi_norm = np.linalg.norm(hi) ** 2
    Mhi = Mi @ hi.T

    if pvals is None:
        pvals = range(1, m, 2)  # Try all possible peak indices

    for p in pvals:
        Up = create_unimodal_matrix(m, p)
        invUp = np.linalg.inv(Up)
        Q = hi_norm * (invUp.T @ invUp)
        if l2 != 0:
            D = create_difference_matrix(m)
            tmp = D @ invUp
            tmp2 = tmp.T @ tmp
            Q = Q + l2 * (np.linalg.norm(Q, 'fro') / np.linalg.norm(tmp2, 'fro')) * tmp2
        if beta != 0:
            tmp3 = invUp.T @ invUp
            Q = Q + beta * (np.linalg.norm(Q, 'fro') / np.linalg.norm(tmp3, 'fro')) * tmp3
        _p = invUp.T @ Mhi
        b = invUp.T @ np.ones((m, 1))

        # Accelerated projected gradient
        ynew = apg(Up @ wi, Q, _p, b)

        score = 0.5 * np.dot((Q @ ynew).T, ynew) - np.dot(_p.T, ynew)
        if score < min_score:
            min_p = p
            min_score = score
            wmin = invUp @ ynew
    return wmin.reshape(m,), min_p

def apg(
    y: np.ndarray,
    Q: np.ndarray,
    _p: np.ndarray,
    b: np.ndarray,
    itermax: int = 100
) -> np.ndarray:
    """
    Accelerated projected gradient method for quadratic programming.

    Returns:
        np.ndarray: Solution vector.
    """
    k = 1
    yhat = ynew = y
    norm_Q = np.linalg.norm(Q, ord=2)
    while (np.linalg.norm(ynew - y) > 1e-3 or k == 1) and k < itermax:
        y = ynew
        z = yhat - (Q @ yhat - _p) / (norm_Q + 1e-16)
        nu = calculate_nu(b, z)
        # Project onto feasible set
        ynew = z - nu * b
        ynew[ynew < 0] = 0
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew

def calculate_nu(b: np.ndarray, z: np.ndarray) -> float:
    """
    Calculate step size for projection in APG.

    Returns:
        float: Step size.
    """
    nz_idx = b >= 0
    nzb = b[nz_idx]
    nzz = z[nz_idx]

    idx = np.argsort(-z / b, 0)
    return np.max((np.cumsum(nzz[idx] * nzb[idx]) - 1) / np.cumsum(nzb[idx] * nzb[idx]))

def update_hi(Mi: np.ndarray, wi: np.ndarray, n: int) -> np.ndarray:
    """
    Update the value of h(i) row as part of BCD.

    Returns:
        np.ndarray: Updated hi row.
    """
    tmp = Mi.T @ wi
    tmp[tmp < 0] = 0
    hi = tmp / (np.linalg.norm(wi) ** 2)
    return hi.reshape(1, n)
