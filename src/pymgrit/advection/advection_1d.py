"""
Vector and application class for the 1D advection problem
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorAdvection1D(Vector):
    """
    Vector class for the 1D advection problem
    """

    def __init__(self, size):
        super(VectorAdvection1D, self).__init__()
        self.size = size
        self.values = np.zeros(size)

    def set_values(self, values):
        self.values = values

    def get_values(self):
        return self.values

    def clone_zero(self):
        return VectorAdvection1D(self.size)

    def clone_rand(self):
        tmp = VectorAdvection1D(self.size)
        tmp.set_values(np.random.rand(self.size))
        return tmp

    def __add__(self, other):
        tmp = VectorAdvection1D(self.size)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorAdvection1D(self.size)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.values)


class Advection1D(Application):
    """
    Application class for the advection problem in 1D space,
       u_t + c * u_t = 0,
    subject to periodic boundary conditions in space and
    initial condition u(x, 0) = exp(-x^2).
    """

    def __init__(self, c, x_start, x_end, nx, *args, **kwargs):
        super(Advection1D, self).__init__(*args, **kwargs)

        self.c = c  # (constant) advection speed

        # spatial domain
        self.x_start = x_start  # lower interval bound of spatial domain
        self.x_end = x_end      # upper interval bound of spatial domain
        self.x = np.linspace(self.x_start, self.x_end, nx)
        # periodic boundary conditions
        self.x = self.x[0:-1]
        self.nx = nx - 1
        # spatial grid spacing
        self.dx = self.x[1] - self.x[0]

        self.identity = identity(self.nx, dtype='float', format='csr')

        # set spatial discretization matrix
        self.space_disc = self.compute_matrix()

        # set initial condition
        self.vector_template = VectorAdvection1D(self.nx)
        self.vector_t_start = VectorAdvection1D(self.nx)
        self.initialise()

    def compute_matrix(self):
        """
        Define spatial discretization matrix for advection problem.

        Discretization is first-order upwind in space.
        """

        fac = self.c / self.dx

        diagonal = np.ones(self.nx) * fac
        lower = np.ones(self.nx) * -fac

        matrix = sp.diags(
            diagonals=[diagonal, lower],
            offsets=[0, -1], shape=(self.nx, self.nx),
            format='lil')
        # set periodic entry
        matrix[0, self.nx - 1] = -fac

        return sp.csr_matrix(matrix)

    def initialise(self):
        """
        Set the initial condition of the 1D advection problem
            u(x,0) = exp(-x^2)
        """
        self.vector_t_start.set_values(np.exp(-self.x ** 2))

    def step(self, u_start: VectorAdvection1D, t_start: float, t_stop: float):
        """
        Time integration routine for 1D advection problem:
            Backward Euler

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution for the input time t_stop
        """
        tmp = u_start.get_values()
        tmp = spsolve((t_stop - t_start) * self.space_disc + self.identity, tmp)
        u_stop = VectorAdvection1D(len(tmp))
        u_stop.set_values(tmp)
        return u_stop
