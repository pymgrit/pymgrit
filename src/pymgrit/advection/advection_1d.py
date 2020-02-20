import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorAdvection1D(Vector):
    """
    Vector for the 1D advection equation
    """

    def __init__(self, size):
        super(VectorAdvection1D, self).__init__()
        self.size = size
        self.values = np.zeros(size)

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

    def clone_zero(self):
        return VectorAdvection1D(self.size)

    def clone_rand(self):
        tmp = VectorAdvection1D(self.size)
        tmp.set_values(np.random.rand(self.size))
        return tmp

    def set_values(self, values):
        self.values = values

    def get_values(self):
        return self.values


class Advection1D(Application):
    """
    Class containing the description of the advection problem.
    """

    def __init__(self, c, x_start, x_end, nx, *args, **kwargs):
        super(Advection1D, self).__init__(*args, **kwargs)

        self.c = c  # advection speed

        # spatial domain
        self.x_start = x_start  # lower interval bound of spatial domain
        self.x_end = x_end  # upper interval bound of spatial domain
        self.x = np.linspace(self.x_start, self.x_end, nx)
        # periodic boundary conditions
        self.x = self.x[0:-1]
        self.nx = nx - 1
        self.dx = self.x[1] - self.x[0]

        self.identity = identity(self.nx, dtype='float', format='csr')

        # set discretization matrix
        self.space_disc = self.compute_matrix()

        # set initial condition
        self.vector_template = VectorAdvection1D(self.nx)
        self.vector_t_start = VectorAdvection1D(self.nx)

        self.initialise()

    def compute_matrix(self):
        """
        Define discretization matrix for advection problem.

        Discretization is first-order upwind in space and
        backward Euler in time.
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
        Initial condition
        """
        # # low frequencies
        # self.vector_t_start.set_values(2 * np.cos((np.pi/16)*self.x))

        # high frequencies
        self.vector_t_start.set_values(2 * np.cos((10 * np.pi / 16) * self.x))

        # # linear combination of low and high frequencies
        # self.vector_t_start.set_values(2 * np.cos((np.pi/8)*self.x) + 2 * np.cos((15 * np.pi / 16) * self.x))

        # Gaussian
        # self.vector_t_start.set_values(np.exp(-self.x ** 2))

    def step(self, u_start: VectorAdvection1D, t_start: float, t_stop: float):
        tmp = u_start.get_values()
        tmp = spsolve((t_stop - t_start) * self.space_disc + self.identity, tmp)
        ret = VectorAdvection1D(len(tmp))
        ret.set_values(tmp)
        return ret
