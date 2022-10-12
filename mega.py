import numpy as np
from scipy.signal import find_peaks

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
        vals = vals.union(range(i - nrad, i + nrad + 1))
    return list(vals.intersection(range(0, m)))


def get_peaks(M, nrad=2):
    """Returns a list containing all integer values in the neighborhoods of likely peaks."""
    (m, n) = M.shape
    all_peaks = list()
    for j in range(n):
        # TODO: find best parameters
        peaks, _ = find_peaks(x=M[:, j].reshape(m, ), prominence=1, width=10)
        all_peaks.extend(peaks)
    return get_neighbors(all_peaks, m, nrad=nrad)


def numf(M, W, H, pvals=None, l2=0, iters=100):
    """Runs the NuMF algorithm to decompose the M vector into unimodal peaks."""
    (m, n) = M.shape
    r = W.shape[1]  # rank

    for it in range(iters):
        for i in range(r):
            wi = W[:, i].reshape(m, 1)
            hi = H[i, :].reshape(1, n)

            Mi = M - W @ H + wi @ hi

            # updating hi
            H[i, :] = update_hi(Mi, wi, n)

            # updating wi
            W[:, i] = update_wi(Mi, hi, m, pvals, l2)
        print(it, np.linalg.norm(M - W @ H, 'fro') / np.linalg.norm(M, 'fro'))
    return W, H


def update_wi(Mi, hi, m, pvals=None, l2=0):
    """Updates the value of w(i) column as part of BCD."""
    wmin = np.empty((m, 1))
    min_score = np.Inf

    if pvals is None:
        pvals = range(1, m, 2)  # trying all p values
    for p in pvals:
        # creating Up matrix
        Up = create_Up(m, p)

        Q = (np.linalg.norm(hi) ** 2) * (np.linalg.inv(Up).T @ np.linalg.inv(Up)) + \
            l2 * ((create_D(m) @ np.linalg.inv(Up)).T @ (create_D(m) @ np.linalg.inv(Up)))
        _p = np.linalg.inv(Up).T @ (Mi @ hi.T)
        b = np.linalg.inv(Up).T @ np.ones((m, 1))

        # accelerated projected gradient
        ynew = apg(Q, _p, b, m)

        score = 0.5 * np.dot((Q @ ynew).T, ynew) - np.dot(_p.T, ynew)
        if score < min_score:
            min_score = score
            wmin = np.linalg.inv(Up) @ ynew
    return wmin.reshape(m, )


def apg(Q, _p, b, m):
    """Runs acceraled projected gradient."""
    k = 1
    yhat = ynew = y = np.random.rand(m, 1)
    while np.linalg.norm(ynew - y) > 1e-8 or k == 1:
        y = ynew
        z = yhat - (Q @ yhat - _p) / (np.linalg.norm(Q, ord=2) + 1e-8)
        idx = np.argsort(z / b, 0)
        # TODO: check nu solution
        # nu = np.max((np.cumsum(z[idx] * b[idx]) - 1) / np.cumsum(b[idx] * b[idx]))
        # TODO: try alternate optimization methods
        nu = np.max((np.cumsum(z * b) - 1) / np.cumsum(b * b))
        ynew = z - nu * b
        ynew[ynew < 0] = 0
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew


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


def main():
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

    W0 = np.random.rand(m, r)
    H0 = np.random.rand(r, n)

    pvals = get_neighbors([p1, p2, p3], m, 1)
    (W, H) = numf(M, W0, H0, pvals, iters=5)


if __name__ == '__main__':
    main()
