import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from abstract_classes import application
from heat_equation import vector_standard


class HeatEquation(application.Application):
    """
    TODO
    """

    def __init__(self, x_start, x_end, nx, *args, **kwargs):
        super(HeatEquation, self).__init__(*args, **kwargs)
        self.x_start = x_start
        self.x_end = x_end
        self.x = np.linspace(self.x_start, self.x_end, nx)
        self.x = self.x[1:-1]
        self.nx = nx - 2

        self.a = self.heat_sparse(np.size(self.x), (1 * (self.t[1] - self.t[0])) / (self.x[1] - self.x[0]) ** 2)

        self.u = [None] * self.nt
        for i in range(self.nt):
            self.u[i] = vector_standard.VectorStandard(self.nx)

        self.u[0].vec = self.u_exact(self.x, 0)

    @staticmethod
    def heat_sparse(nx, fac):
        """
        TODO
        """
        diagonal = np.zeros(nx)
        lower = np.zeros(nx - 1)
        upper = np.zeros(nx - 1)

        diagonal[:] = 1 + 2 * fac
        lower[:] = -fac
        upper[:] = -fac

        a = sp.diags(
            diagonals=[diagonal, lower, upper],
            offsets=[0, -1, 1], shape=(nx, nx),
            format='csr')

        return sp.csc_matrix(a)

    @staticmethod
    def u_exact(x, t):
        """
        TODO
        """
        # return x * (x - 1) * np.sin(2 * np.pi * t)
        return np.sin(np.pi * x) * np.cos(t)

    @staticmethod
    def f(x, t):
        """
        TODO
        """
        # return 2 * np.pi * x * (x - 1) * np.cos(2 * np.pi * t) - 2 * np.sin(2 * np.pi * t)
        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def u_exact_complete(self, x, t):
        """
        TODO
        """
        ret = np.zeros((np.size(t), np.size(x)))
        for i in range(np.size(t)):
            ret[i] = self.u_exact(x, t[i])
        return ret

    def step(self, index):
        u_start = self.u[index-1]
        t_start = self.t[index-1]
        t_stop = self.t[index]
        tmp = u_start.vec
        tmp = spsolve(self.a, tmp + self.f(self.x, t_stop) * (t_stop - t_start))
        ret = vector_standard.VectorStandard(u_start.size)
        ret.vec = tmp
        return ret
