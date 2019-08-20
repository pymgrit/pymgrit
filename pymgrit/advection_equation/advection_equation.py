import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from pymgrit.core import application
from pymgrit.advection_equation import vector_standard


class AdvectionEquation(application.Application):
    """
    Class containing the description of the advection problem.
    """

    def __init__(self, c, x_start, x_end, nx, *args, **kwargs):

        super(AdvectionEquation, self).__init__(*args, **kwargs)

        self.c = c  # advection speed

        # spatial domain
        self.x_start = x_start  # lower interval bound of spatial domain
        self.x_end = x_end  # upper interval bound of spatial domain
        self.x = np.linspace(self.x_start, self.x_end, nx)
        # periodic boundary conditions
        self.x = self.x[0:-1]
        self.nx = nx - 1

        # set discretization matrix
        self.a = self.advection_sparse(self.nx, (c * (self.t[1] - self.t[0])) / (self.x[1] - self.x[0]))

        # set initial condition
        self.u = vector_standard.VectorStandard(self.nx)
        self.initialise()

    @staticmethod
    def advection_sparse(nx, fac):
        """
        Define discretization matrix for advection problem.

        Discretization is first-order upwind in space and
        backward Euler in time.
        """
        diagonal = np.zeros(nx)
        lower = np.zeros(nx - 1)

        diagonal[:] = 1 + fac
        lower[:] = -fac

        a = sp.diags(
            diagonals=[diagonal, lower],
            offsets=[0, -1], shape=(nx, nx),
            format='lil')
        # set periodic entry
        a[0, nx - 1] = -fac

        return sp.csc_matrix(a)

    def initialise(self):
        """
        Initial condition
        """
        # # low frequencies
        # self.u.vec = 2 * np.cos((np.pi/16)*self.x)

        # high frequencies
        self.u.vec = 2 * np.cos((10*np.pi / 16) * self.x)

        # # linear combination of low and high frequencies
        # self.u.vec = 2 * np.cos((np.pi/8)*self.x) + 2 * np.cos((15 * np.pi / 16) * self.x)

        # Gaussian
        #self.u.vec = np.exp(-self.x ** 2)

    def step(self, u_start, t_start, t_stop):
        tmp = u_start.vec
        tmp = spsolve(self.a, tmp)
        ret = vector_standard.VectorStandard(u_start.size)
        ret.vec = tmp
        return ret
