import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

np.random.seed(42)


def get_peaks(M, nrad=2):
    (m, n) = M.shape
    all_peaks = list()
    for j in range(n):
        peaks, _ = find_peaks(x=M[:, j].reshape(m, ), prominence=1, width=10)
        all_peaks.extend(peaks)
    return get_neighbors(all_peaks, m, nrad=nrad)


def numf(M, W, H, iters=100, peak_vals=None):
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
            W[:, i] = update_wi(Mi, hi, m, peak_vals=peak_vals)
            print(i)
        print(it, np.linalg.norm(M - W @ H, 'fro') / np.linalg.norm(M, 'fro'))
    return W, H


def update_wi(Mi, hi, m, peak_vals=None):
    wmin = np.empty((m, 1))
    min_score = np.Inf
    if peak_vals is None:
        peak_vals = range(1, m, 2) # trying all p values
    for p in peak_vals:
        # creating Up matrix
        Up = create_Up(m, p)

        # TODO: add regularization (andersen's email)
        Q = (np.linalg.norm(hi) ** 2) * (np.linalg.inv(Up).T @ np.linalg.inv(Up))
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
    k = 1
    yhat = ynew = y = np.random.rand(m, 1)
    diff = np.linalg.norm(ynew - y)
    while diff > 1e-6 or k == 1:
        y = ynew
        z = yhat - (Q @ yhat - _p) / (np.linalg.norm(Q, ord=2) + 1e-6)
        ynew = proj(b, z)
        # TODO: alternate optimization methods
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew


def proj(b, z):
    tmp = b != 0
    b = b[tmp]
    z = z[tmp]

    idx = np.argsort(-z / b, 0)
    nu = np.max((np.cumsum(z[idx] * b[idx]) - 1) / np.cumsum(b[idx] * b[idx]))
    ynew = z - nu * b
    ynew[ynew < 0] = 0
    return ynew.reshape(-1, 1)


def create_Up(m, p):
    D = np.diag(np.ones(p + 1)) + np.diag(-1 * np.ones(p), -1)
    if p < m - 1:
        Dt = np.diag(np.ones(m - p - 1)) + np.diag(-1 * np.ones(m - p - 2), 1)
        Up = np.block([[D, np.zeros((p + 1, m - p - 1))],
                       [np.zeros((m - p - 1, p + 1)), Dt]])
    else:
        Up = D
    return Up


def update_hi(Mi, wi, n):
    tmp = Mi.T @ wi
    tmp[tmp < 0] = 0
    hi = tmp / (np.linalg.norm(wi) ** 2)
    return hi.reshape(1, n)


def main():
    # def gaussian(x, r, alpha):
    #     return 1. / (math.sqrt(alpha ** math.pi)) * np.exp(-alpha * np.power((x - r), 2.))
    # x = np.linspace(0, 10, 20)
    # y1 = gaussian(x, 2, 3)
    # y2 = gaussian(x, 3, 8)
    # y3 = gaussian(x, 2, 5)
    #
    # M = np.array([y1 + y2, y3]).T
    # # M = M.reshape(-1, 1)
    df = pd.read_csv('cases.csv')

    M = df['cases'].to_numpy().reshape(-1, 1)
    r = 16
    m, n = M.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    (W, H) = numf(M, W, H, peak_vals=get_peaks(M, nrad=2), iters=10)

    # plt.plot(x, W)


def get_neighbors(locs, m, nrad=2):
    vals = set()
    for i in locs:
        vals = vals.union(range(i - nrad, i + nrad + 1))
    return vals.intersection(range(0, m))


if __name__ == '__main__':
    main()
