import time
import numpy as np
from scipy import linalg as la
import heatEquation
from scipy.sparse.linalg import spsolve
import mgrit
import helper


def main(L, nt, nx, m, it=15, tol=1e-7):
    # Setup system
    t = np.linspace(0, 5, nt)

    x = np.linspace(0, 1, nx)
    x = x[1:-1]

    res = np.zeros(it + 1)

    u = [[] for i in range(L)]
    v = [[] for i in range(L)]
    g = [[] for i in range(L)]
    tc = [[] for i in range(L)]

    for l in range(L):
        tc[l] = t[np.array(range(0, np.size(t), m ** l))]
        u[l] = np.zeros((np.size(tc[l]), np.size(x)))
        g[l] = np.zeros_like(u[l])
        v[l] = np.zeros_like(u[l])

    app = heatEquation.setup(L, x, tc)
    r = np.zeros_like(u[0])

    def Phi(ustart, tstart, tstop, app=None):
        return spsolve(app['A'], ustart + heatEquation.f(app['x'], tstop) * (tstop - tstart))

    u[0][0] = heatEquation.uExact(x, 0)

    # Residual
    for i in range(1, np.size(tc[0])):
        r[i] = Phi(u[0][i - 1], tc[0][i - 1], tc[0][i], app[0]) - u[0][i]

    res[0] = la.norm(r)
    print("step ", 0, "| norm r=", res[0])

    start = time.time()
    for iter in range(it):

        u, v, g = mgrit.mgrit(0, L, u, v, g, m, tc, Phi, app)

        # Residual
        for i in range(1, np.size(tc[0])):
            r[i] = Phi(u[0][i - 1], tc[0][i - 1], tc[0][i], app[0]) - u[0][i]

        res[iter + 1] = la.norm(r)

        print("step ", iter + 1, "| norm r=", res[iter + 1])
        if res[iter + 1] < tol:
            break

    end = time.time()
    helper.plot1D(x, t, u[0])
    print("time", end - start)
    return u


if __name__ == '__main__':
    u = main(L=2, nt=101, nx=17, m=2, it=15, tol=1e-7)
