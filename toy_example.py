import numpy as np
from algo import numf
from peaks import get_neighbors


def gauss(x, sigma=1, mean=0, scale=1):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))


def toy_example():


    m = 100
    r = 3
    p1 = 24
    p2 = 50
    p3 = 76

    x = np.linspace(1, m, m).reshape(-1, 1)
    w1 = gauss(x, sigma=2, mean=p1)
    w2 = np.concatenate((np.zeros((int((m - p2) / 2), 1)), np.ones((p2, 1)), np.zeros((int((m - p2) / 2), 1))))
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
    r = 3
    M, Wtrue, Htrue, ptrue = toy_example()
    m, n = M.shape

    W0 = np.random.rand(m, r)
    H0 = np.random.rand(r, n)

    pvals = get_neighbors(ptrue, m, nrad=2)
    W, H, _ = numf(M, W0, H0, pvals=pvals, iters=20)


if __name__ == '__main__':
    main()