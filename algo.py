import numpy as np
from scipy.signal import find_peaks
import pandas as pd

np.random.seed(42)


def create_D(m):
    """Creates a (m - 1, m) size first order difference matrix."""
    D = np.zeros((m - 1, m))
    i, j = np.indices(D.shape)
    D[i == j] = -1
    D[i == j - 1] = 1
    return D


def get_neighbors(locs, m, nrad=2):
    """Returns a list containing integers close to integers in locs list."""
    vals = set()
    for i in locs:
        vals = vals.union(range(i - nrad, i + nrad + 1, 2))
    return list(vals.intersection(range(0, m)))


def get_peaks(M, nrad=2):
    """Returns a list containing all integer values in the neighborhoods of likely peaks."""
    (m, n) = M.shape
    all_peaks = list()
    for j in range(n):
        # TODO: find best parameters
        peaks, _ = find_peaks(x=M[:, j].reshape(m, ), prominence=1, width=6)
        all_peaks.extend(peaks)
    return get_neighbors(all_peaks, m, nrad=nrad)


def numf(M, W, H, pvals=None, l2=0, iters=10, save_file=None):
    """Runs the NuMF algorithm to decompose the M vector into unimodal peaks."""
    (m, n) = M.shape
    r = W.shape[1]  # rank

    for it in range(iters):
        pouts = list()
        for i in range(r):
            wi = W[:, i].reshape(m, 1)
            hi = H[i, :].reshape(1, n)

            Mi = M - W @ H + wi @ hi

            # updating hi
            H[i, :] = update_hi(Mi, wi, n)

            # updating wi
            W[:, i], pout = update_wi(Mi, hi, m, pvals, l2)
            pouts.append(pout)
        print(it, np.linalg.norm(M - W @ H, 'fro') / np.linalg.norm(M, 'fro'))
        if save_file is not None:
            with open(save_file, 'wb') as fout:
                np.savez_compressed(fout, W=W, H=H)
            print(f'W and H matrices saved in {save_file}.')
    return W, H, pouts


def update_wi(Mi, hi, m, pvals=None, l2=0):
    """Updates the value of w(i) column as part of BCD."""
    wmin = np.empty((m, 1))
    min_score = np.Inf
    min_p = 0

    if pvals is None:
        pvals = range(1, m, 2)  # trying all p values

    for p in pvals:
        # creating Up matrix
        Up = create_Up(m, p)
        invUp = np.linalg.inv(Up)
        Q = (np.linalg.norm(hi) ** 2) * (invUp.T @ invUp)
        if l2 != 0:
            D = create_D(m)
            tmp = D @ invUp
            tmp2 = tmp.T @ tmp
            Q = Q + l2 * (np.linalg.norm(Q, 'fro') / np.linalg.norm(tmp2, 'fro')) * tmp2
        _p = invUp.T @ (Mi @ hi.T)
        b = invUp.T @ np.ones((m, 1))

        # accelerated projected gradient
        ynew = apg(Q, _p, b, m)

        score = 0.5 * np.dot((Q @ ynew).T, ynew) - np.dot(_p.T, ynew)
        if score < min_score:
            min_p = p
            min_score = score
            wmin = invUp @ ynew
    return wmin.reshape(m, ), min_p


def apg(Q, _p, b, m, itermax=100):
    """Runs acceraled projected gradient."""
    k = 1
    yhat = ynew = y = np.random.rand(m, 1)
    while (np.linalg.norm(ynew - y) > 1e-3 or k == 1) and k < itermax:  # temporary
        y = ynew
        z = yhat - (Q @ yhat - _p) / (np.linalg.norm(Q, ord=2) + 1e-8)
        nu = calculate_nu(b, z)
        # TODO: try alternate optimization methods
        ynew = z - nu * b
        ynew[ynew < 0] = 0
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew


def calculate_nu(b, z):
    nz_idx = b >= 0
    nzb = b[nz_idx]
    nzz = z[nz_idx]

    idx = np.argsort(-nzz / nzb, 0)
    return np.max((np.cumsum(nzz[idx] * nzb[idx]) - 1) / np.cumsum(nzb[idx] * nzb[idx]))


def create_Up(m, p):
    """Creates unimodal restriction matrix."""
    D = np.diag(np.ones(p + 1)) + np.diag(-1 * np.ones(p), -1)
    if p < m - 1:
        Dt = np.diag(np.ones(m - p - 1)) + np.diag(-1 * np.ones(m - p - 2), 1)
        Up = np.block([[D, np.zeros((p + 1, m - p - 1))],
                       [np.zeros((m - p - 1, p + 1)), Dt]])
    else:
        Up = D
    return Up


def update_hi(Mi, wi, n):
    """Updates the value of h(i) row as part of BCD."""
    tmp = Mi.T @ wi
    tmp[tmp < 0] = 0
    hi = tmp / (np.linalg.norm(wi) ** 2)
    return hi.reshape(1, n)


def toy_example():
    def gauss(x, sigma=1, mean=0, scale=1):
        return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

    m = 50
    r = 3
    p1 = 12
    p2 = 25
    p3 = 38

    x = np.linspace(1, m, m).reshape(-1, 1)
    w1 = gauss(x, sigma=2, mean=p1)
    w2 = np.concatenate((np.zeros((int((m - 25) / 2), 1)), np.ones((p2, 1)), np.zeros((int((m - 25) / 2) + 1, 1))))
    w3 = gauss(x, sigma=2, mean=p3)
    Wtrue = np.hstack((w1, w2, w3))

    n = 6
    c = 1 / np.sqrt(r - 1)
    e = 0.001
    Htrue = np.array([[c + e, 1 - c - e, 0],
                      [1 - c - e, c + e, 0],
                      [c + e, 0, 1 - c - e],
                      [1 - c - e, 0, c + e],
                      [0, c + e, 1 - c - e],
                      [0, 1 - c - e, c + e]]).T

    M = Wtrue @ Htrue
    ptrue = [p1, p2, p3]
    return M, Wtrue, Htrue, ptrue


def main():
    # df = pd.read_csv('data/cases.csv')
    # M = df['cases'].to_numpy().reshape(-1, 1)
    r = 3
    M, Wtrue, Htrue, ptrue = toy_example()

    m, n = M.shape
    W0 = np.random.rand(m, r)
    H0 = np.random.rand(r, n)

    # pvals = get_peaks(M, nrad=2)
    pvals = get_neighbors(ptrue, m, nrad=2)
    W, H, _ = numf(M, W0, H0, pvals=pvals, l2=0.1, iters=50)


if __name__ == '__main__':
    main()
