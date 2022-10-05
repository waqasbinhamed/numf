import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)


def numf(M, W, H, iters=100):
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
            W[:, i] = update_wi(Mi, hi, m, n)
        print(it, np.linalg.norm(M - W @ H, 'fro')/np.linalg.norm(M, 'fro'))
    return W, H


def update_wi(Mi, hi, m, n):
    wmin = np.empty((m, 1))
    min_score = np.Inf
    for p in range(m):  # trying all p values
        # creating Up matrix
        Up = create_Up(m, p)

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
    while np.linalg.norm(ynew - y) > 1e-8 or k == 1:
        y = ynew
        z = yhat - (Q @ yhat - _p) / np.linalg.norm(Q, ord=2)
        idx = np.argsort(z / b, 0)
        # nu = np.max((np.cumsum(z[idx] * b[idx]) - 1) / np.cumsum(b[idx] * b[idx]))
        nu = np.max((np.cumsum(z * b) - 1) / np.cumsum(b * b))
        ynew = z - nu * b
        ynew[ynew < 0] = 0
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew


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
    def gaussian(x, r, alpha):
        return 1. / (math.sqrt(alpha ** math.pi)) * np.exp(-alpha * np.power((x - r), 2.))

    x = np.linspace(0, 10, 100)
    y1 = gaussian(x, 2, 5)
    y2 = gaussian(x, 4, 8)
    y2 = gaussian(x, 2, 1)

    M = y1 + y2
    M = M.reshape(-1, 1)

    r = 3
    m, n = M.shape
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    (W, H) = numf(M, W, H, iters=100)

    # plt.plot(x, W)


if __name__ == '__main__':
    main()
