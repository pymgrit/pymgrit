"""
Heat equation 2-d example based on
'The Craft of Finite Difference Computing with Partial Differential Equations' from H. P. Langtangen
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorHeat2D(Vector):
    """
    Vector for the 2D heat equation
    """

    def __init__(self, nx, ny):
        super(VectorHeat2D, self).__init__()
        self.nx = nx
        self.ny = ny
        self.values = np.zeros((self.nx, self.ny))

    def __add__(self, other):
        tmp = VectorHeat2D(self.nx, self.ny)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorHeat2D(self.nx, self.ny)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.values)

    def clone_zero(self):
        return VectorHeat2D(self.nx, self.ny)

    def clone_rand(self):
        tmp = VectorHeat2D(self.nx, self.ny)
        tmp.set_values(np.random.rand((self.nx, self.ny)))
        return tmp

    def set_values(self, values):
        self.values[1:-1, 1:-1] = values

    def get_values(self):
        return self.values[1:-1, 1:-1]


class Heat2D(Application):
    """
    Heat equation 2-d example
    """

    def __init__(self, lx, ly, nx, ny, a, *args, **kwargs):
        super(Heat2D, self).__init__(*args, **kwargs)

        self.nx = nx
        self.ny = ny
        self.x = np.linspace(0, lx, self.nx)
        self.y = np.linspace(0, ly, self.ny)
        self.x_b = self.x[1:-1]
        self.y_b = self.y[1:-1]
        self.nx_b = len(self.x_b)
        self.ny_b = len(self.y_b)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.a = a
        self.lx = lx
        self.ly = ly
        self.x2 = self.x_b[:, np.newaxis]
        self.y2 = self.y_b[np.newaxis, :]

        self.space_disc = self.compute_matrix()
        self.identity = identity(self.nx_b * self.ny_b, dtype='float', format='csr')

        self.vector_template = VectorHeat2D(self.nx, self.ny)
        self.vector_t_start = VectorHeat2D(self.nx, self.ny)  # Create initial value solution
        self.vector_t_start.set_values(np.zeros((self.nx_b, self.ny_b)))  # Set initial value

    def u_exact(self, x, y, t):
        return 5 * t * x * (self.lx - x) * y * (self.ly - y)

    def i(self, x, y):
        return self.u_exact(x, y, 0)

    def f(self, x, y, t):
        return 5 * x * (self.lx - x) * y * (self.ly - y) + 10 * self.a * t * (y * (self.ly - y) + x * (self.lx - x))

    def compute_rhs(self, u_start, t_start, t_stop):
        return u_start.T.flatten() + (t_stop - t_start) * self.f(self.x2, self.y2, t_stop).T.flatten()

    def compute_matrix(self):
        n = self.nx_b * self.ny_b
        fx = (self.a / self.dx ** 2)
        fy = (self.a / self.dy ** 2)

        main = np.ones(n) * (2 * (fx + fy))
        upper = np.ones(n - 1) * -fx
        upper[(self.nx_b - 1)::self.nx_b] = 0
        lower = np.ones(n - 1) * -fx
        lower[(self.nx_b - 1)::self.nx_b] = 0
        upper2 = np.ones(n - self.nx_b) * -fy
        lower2 = np.ones(n - self.nx_b) * -fy

        matrix = diags(diagonals=[main, lower, upper, lower2, upper2],
                       offsets=[0, -1, 1, -self.nx_b, self.nx_b],
                       shape=(n, n), format='csr')

        return matrix

    def step(self, u_start: VectorHeat2D, t_start: float, t_stop: float) -> VectorHeat2D:
        """
        Forward-difference in time
        :param u_start:
        :param t_start:
        :param t_stop:
        :return:
        """
        rhs = self.compute_rhs(u_start=u_start.get_values(), t_start=t_start, t_stop=t_stop)
        new = spsolve((t_stop - t_start) * self.space_disc + self.identity, rhs)
        ret = VectorHeat2D(self.nx, self.ny)
        ret.set_values(new.reshape(self.ny_b, self.nx_b).T)
        return ret
