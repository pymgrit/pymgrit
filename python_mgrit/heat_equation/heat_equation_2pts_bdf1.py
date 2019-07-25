import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from abstract_classes import application
from heat_equation import vector_standard_bdf2
import math


class HeatEquation(application.Application):
    """
    """

    def __init__(self, x_start, x_end, nx, dt, *args, **kwargs):
        super(HeatEquation, self).__init__(*args, **kwargs)
        self.x_start = x_start
        self.x_end = x_end
        self.x = np.linspace(self.x_start, self.x_end, nx)
        self.x = self.x[1:-1]
        self.nx = nx - 2
        self.dt = dt

        self.u_ex = self.u_exact_complete(x=self.x, t=np.linspace(self.t_start, self.t_end, (self.nt - 1) * 2 + 1))

        self.a1 = self.heat_sparse(np.size(self.x),
                                   (1 * (self.t[1] - self.t[0] - self.dt)) / (self.x[1] - self.x[0]) ** 2)
        self.a2 = self.heat_sparse(np.size(self.x), (1 * self.dt) / (self.x[1] - self.x[0]) ** 2)

        self.u = vector_standard_bdf2.VectorStandardBDF2(self.nx)

        self.u.vec_first_time_point = self.u_exact(self.x, self.t[0])
        self.u.vec_second_time_point = spsolve(self.a2,
                                               self.u.vec_first_time_point + self.f(self.x, self.t[0] + dt) * self.dt)

    @staticmethod
    def heat_sparse(nx, fac):
        """
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
        """
        # return x * (x - 1) * np.sin(2 * np.pi * t)
        return np.sin(np.pi * x) * np.cos(t)

    @staticmethod
    def f(x, t):
        """
        """
        # return 2 * np.pi * x * (x - 1) * np.cos(2 * np.pi * t) - 2 * np.sin(2 * np.pi * t)
        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def u_exact_complete(self, x, t):
        """
        """
        ret = np.zeros((np.size(t), np.size(x)))
        for i in range(np.size(t)):
            ret[i] = self.u_exact(x, t[i])
        return ret

    def step(self, u_start, t_start, t_stop):
        tmp1 = spsolve(self.a1, u_start.vec_second_time_point + self.f(self.x, t_stop) * (t_stop - t_start - self.dt))

        tmp2 = spsolve(self.a2, tmp1 + self.f(self.x, t_stop + self.dt) * self.dt)

        ret = vector_standard_bdf2.VectorStandardBDF2(u_start.size)
        ret.vec_first_time_point = tmp1
        ret.vec_second_time_point = tmp2
        return ret
