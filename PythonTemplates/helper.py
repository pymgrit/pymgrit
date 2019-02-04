from operator import itemgetter
from itertools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.sparse import block_diag
from scipy.sparse import diags
from scipy import sparse as sp


def computeDistribution(length, size, rank):
    blockSize = splitInto(length, size)[rank]

    firstI = 0
    if blockSize > 0:
        for i in range(size):
            if i == rank:
                break
            firstI += splitInto(length, size)[i]
    return blockSize, firstI


def splitInto(n, p):
    return np.array([int(n / p + 1)] * (n % p) + [int(n / p)] * (p - n % p))


def setupDistribution(nt, m, size, rank):
    allPts = np.array(range(0, nt))
    blockSize, firstI = computeDistribution(np.size(allPts), size, rank)
    allPts = allPts[firstI:firstI + blockSize]

    allCpts = np.array(range(0, nt, m))
    allFpts = np.array(list(set(np.array(range(0, nt))) - set(allCpts)))

    cpts = np.sort(np.array(list(set(allPts) - set(allFpts)), dtype='int'))
    blockSize = np.size(cpts)
    if blockSize > 0:
        firstI = np.where(cpts[0] == allCpts)[0][0]
    else:
        firstI = 0
    fpts = np.array(list(set(allPts) - set(cpts)))

    fpts2 = np.array([item for sublist in np.array([np.array(xi) for xi in np.asarray(
        [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(fpts), lambda x: x[0] - x[1])])])[::-1] for item in
                      sublist])

    allCpts = np.array(range(0, nt, m))
    allFpts = np.array(list(set(np.array(range(0, nt))) - set(allCpts)))
    needCommunicationFront = False
    needCommunicationBack = False

    if np.size(fpts) > 0 and fpts[np.argmin(fpts)] - 1 in allFpts:
        needCommunicationFront = True
    if np.size(fpts) > 0 and fpts[np.argmax(fpts)] + 1 in allFpts:
        needCommunicationBack = True

    return fpts2, cpts, allCpts, allPts, firstI, blockSize, needCommunicationFront, needCommunicationBack


def plot1D(x, y, U):
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, U, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('spatial')
    ax.set_ylabel('time')
    ax.set_zlabel('temperature')
    plt.show()
