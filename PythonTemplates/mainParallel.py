from mpi4py import MPI
import time
import numpy as np
from scipy import linalg as la
import heatEquation
import mgritParallel
import helper
from scipy.sparse.linalg import spsolve


def main(L, nt, nx, m, it=15, tol=1e-7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
        tc[l] = t[np.array(range(0, np.size(t), m ** (l)))]
        u[l] = np.zeros((np.size(tc[l]), np.size(x)))
        g[l] = np.zeros_like(u[l])
        v[l] = np.zeros_like(u[l])

    app = heatEquation.setup(L, x, tc)

    def Phi(ustart, tstart, tstop, app=None):
        return spsolve(app['A'], ustart + heatEquation.f(app['x'], tstop) * (tstop - tstart))

    blockSize, firstI = helper.computeDistribution(np.size(t) - 1, size, rank)

    u[0][0] = heatEquation.uExact(x,0)

    r = np.zeros((blockSize, np.size(u[0], 1)))

    for i in range(firstI + 1, (blockSize + firstI + 1)):
        j = i - firstI - 1
        r[j] = Phi(u[0][i - 1], tc[0][i - 1], tc[0][i], app[0]) - u[0][i]

    rFull = comm.gather(r, root=0)

    if rank == 0:
        rFull = np.vstack(rFull)
        rFull = np.vstack((np.zeros_like(rFull[0]), rFull))
        res[0] = la.norm(rFull)
        print("step  0 | norm r=", res[0])

    start = time.time()
    for iters in range(it):

        u, v, g = mgritParallel.mgrit(0, L, u, v, g, m, tc, Phi, app)

        u = comm.bcast(u, root=0)

        r = np.zeros((blockSize, np.size(u[0], 1)))
        for i in range(firstI + 1, (blockSize + firstI + 1)):
            j = i - firstI - 1
            if i != 0:
                r[j] = Phi(u[0][i - 1], tc[0][i - 1], tc[0][i], app[0]) - u[0][i]

        rFull = comm.gather(r, root=0)

        if rank == 0:
            rFull = np.vstack(rFull)
            rFull = np.vstack((np.zeros_like(rFull[0]), rFull))
            res[iters + 1] = la.norm(rFull)
            print("step ", iters + 1, "| norm r=", res[iters + 1])

        res[iters + 1] = comm.bcast(res[iters + 1], root=0)

        if res[iters + 1] < tol:
            break

    end = time.time()
    if rank == 0:
        print("time", end - start)
        # helper.plot1D(x, t, u[0])
    return u


if __name__ == '__main__':
    u = main(L=2, nt=101, nx=17, m=2, it=15, tol=1e-7)
