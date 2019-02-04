import numpy as np
from mpi4py import MPI
import helper
import sys


def mgrit(l, L, u, v, g, m, t, Phi, app):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Solve problem on coarsest grid
    if l == L - 1:
        if rank == 0:
            for i in range(1, np.size(t[l])):
                u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])
        u = comm.bcast(u, root=0)
        return u, v, g

    # Compute F and C points
    nt = np.size(t[l])
    fpts, cpts, cptsCom, allpts, firstI, blockSize, commFront, commBack = helper.setupDistribution(nt, m, size, rank)

    # FCF-relaxation
    if np.size(fpts) > 0:
        for i in np.nditer(fpts):
            if i == np.min(fpts) and commFront:
                req = comm.Irecv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)
                req.Wait()
            u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])
            if i == np.max(fpts) and commBack:
                comm.Isend(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    if (allpts.shape[0] > 0 and allpts[0] > 0 and np.size(np.where(cpts == allpts[0])[0]) == 1):
        comm.Recv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)
    if allpts.shape[0] > 0 and allpts[-1] < nt - 1 and np.size(np.where(fpts == allpts[-1])[0]) == 1 and np.size(
            np.where(cptsCom == allpts[-1] + 1)[0]) == 1:
        comm.Send(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    if np.size(cpts) > 0:
        for i in np.nditer(cpts):
            if i != 0:
                u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

    if allpts.shape[0] > 0 and allpts[0] > 0 and np.size(np.where(fpts == allpts[0])[0]) == 1 and np.size(
            np.where(cptsCom == allpts[0] - 1)[0]) == 1:
        comm.Recv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)

    if allpts.shape[0] > 0 and allpts[-1] < nt - 1 and np.size(np.where(cpts == allpts[-1])[0]) == 1:
        comm.Send(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    if np.size(fpts) > 0:
        for i in np.nditer(fpts):
            if i == np.min(fpts) and commFront:
                req = comm.Irecv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)
                req.Wait()

            u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

            if i == np.max(fpts) and commBack:
                comm.Isend(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    u[l] = np.vstack(
        comm.allgather(u[l][allpts[0]:(allpts[-1] + 1), :] if np.size(allpts) > 0 else np.empty((0, u[l].shape[1]))))

    v[l + 1] = np.copy(u[l][cptsCom, :])

    if np.size(cpts) > 0:
        for i in range(0, np.size(cpts)):
            j = i + firstI
            if j != 0:
                g[l + 1][i] = g[l][cpts[i]] \
                              + Phi(u[l][cpts[i] - 1], t[l][cpts[i] - 1], t[l][cpts[i]], app[l]) - u[l][cpts[i]] \
                              - Phi(v[l + 1][j - 1], t[l + 1][j - 1], t[l + 1][j], app[l + 1]) + v[l + 1][j]

    g[l + 1] = np.vstack(comm.allgather(g[l + 1][0:np.size(cpts), :]))

    u[l + 1] = np.copy(v[l + 1])

    u, v, g = mgrit(l + 1, L, u, v, g, m, t, Phi, app)

    e = u[l + 1][firstI:firstI + blockSize] - v[l + 1][firstI:firstI + blockSize]

    u[l][cpts] = u[l][cpts] + e

    if allpts.shape[0] > 0 and allpts[0] > 0 and np.size(np.where(fpts == allpts[0])[0]) == 1 and np.size(
            np.where(cptsCom == allpts[0] - 1)[0]) == 1:
        comm.Recv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)
    if allpts.shape[0] > 0 and allpts[-1] < nt - 1 and np.size(np.where(cpts == allpts[-1])[0]) == 1:
        comm.Send(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    if np.size(fpts) > 0:
        for i in np.nditer(fpts):
            if i == np.min(fpts) and commFront:
                req = comm.Irecv(u[l][allpts[0] - 1], rank - 1, tag=allpts[0] - 1)
                req.Wait()
            u[l][i] = g[l][i] + Phi(u[l][i - 1], t[l][i - 1], t[l][i], app[l])

            if i == np.max(fpts) and commBack:
                comm.Isend(u[l][allpts[-1]], rank + 1, tag=allpts[-1])

    u[l] = np.vstack(
        comm.allgather(u[l][allpts[0]:(allpts[-1] + 1), :] if np.size(allpts) > 0 else np.empty((0, u[l].shape[1]))))

    return u, v, g
