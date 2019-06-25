import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from application import application


class HeatEquation(application.Application):
    """
    TODO
    """

    def __init__(self, x_start, x_end, *args, **kwargs):
        super(HeatEquation, self).__init__(*args, **kwargs)
        self.x_start = x_start
        self.x_end = x_end
        self.x = np.linspace(self.x_start, self.x_end, self.nx[0])
        self.x = self.x[1:-1]
        self.nx[0] = len(self.x)

    def setup(self, lvl_max, t, spatial_coarsening):
        """

        :rtype: object
        """
        a = [None] * lvl_max
        app = [None] * lvl_max

        for l in range(lvl_max):
            a[l] = self.heat_sparse(np.size(self.x), (1 * (t[l][1] - t[l][0])) / (self.x[1] - self.x[0]) ** 2)
            app[l] = {'A': a[l], 'x': self.x}

        return app

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

    def initial_value(self):
        """

        :rtype: object
        """
        return self.u_exact(self.x, 0)

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

    def phi(self, u_start, t_start, t_stop, app):
        return spsolve(app['A'], u_start + self.f(app['x'], t_stop) * (t_stop - t_start))

    def restriction(self, u, app=None):
        pass

    def interpolation(self, u, app=None):
        pass

    def info(self):
        return 'heat_equation/t-[' + str(self.t_start) + ';' + str(self.t_end) + ']/nt-' + str(
            self.nt) + '/'
