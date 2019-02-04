import numpy as np
from scipy import sparse as sp


def setup(sC, x, t):
    A = [[] for i in range(sC)]
    app = [[] for i in range(sC)]

    for l in range(sC):
        A[l] = heatSparse(np.size(x), (1 * (t[l][1] - t[l][0])) / (x[1] - x[0]) ** 2)
        app[l] = {'A': A[l], 'x': x}

    return app


def heatSparse(nx, fac):
    diagonal = np.zeros(nx)
    lower = np.zeros(nx - 1)
    upper = np.zeros(nx - 1)

    diagonal[:] = 1 + 2 * fac
    lower[:] = -fac
    upper[:] = -fac

    A = sp.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(nx, nx),
        format='csr')

    return sp.csc_matrix(A)


def uExact(x, t):
    # return x * (x - 1) * np.sin(2 * np.pi * t)
    return np.sin(np.pi * x) * np.cos(t)


def f(x, t):
    # return 2 * np.pi * x * (x - 1) * np.cos(2 * np.pi * t) - 2 * np.sin(2 * np.pi * t)
    return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))


def uExactAll(x, t):
    ret = np.zeros((np.size(t), np.size(x)))
    for i in range(np.size(t)):
        ret[i] = uExact(x, t[i])
    return ret
