import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

from pymgrit.core.application import Application
from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts


class Heat1DBDF2(Application):
    """
    Application class for the heat equation in 1D space,
        u_t - a*u_xx = b(x,t),  a > 0, x in [x_start,x_end], t in [0,T],
    """

    def __init__(self, x_start, x_end, nx, dt, a, u_exact, rhs, *args, **kwargs):
        super(Heat1DBDF2, self).__init__(*args, **kwargs)
        self.x_start = x_start
        self.x_end = x_end
        self.x = np.linspace(self.x_start, self.x_end, nx)
        self.x = self.x[1:-1]
        self.nx = nx - 2
        self.dt = dt
        self.a = a
        self.dx = self.x[1] - self.x[0]
        self.identity = identity(self.nx, dtype='float', format='csr')
        self.u_exact = u_exact
        self.rhs = rhs

        # set spatial discretization matrix
        self.space_disc = self.compute_matrix()

        self.vector_template = VectorHeat1D2Pts(self.nx)  # Create initial value solution
        self.vector_t_start = VectorHeat1D2Pts(self.nx)
        self.vector_t_start.set_values(first_time_point=self.u_exact(self.x, self.t[0]),
                                       second_time_point=self.u_exact(self.x, self.t[0] + self.dt))

    def compute_matrix(self):
        """
        Space discretization
        """

        fac = self.a / self.dx ** 2

        diagonal = np.ones(self.nx) * (4 / 3) * fac
        lower = np.ones(self.nx - 1) * -(2 / 3) * fac
        upper = np.ones(self.nx - 1) * -(2 / 3) * fac

        matrix = sp.diags(
            diagonals=[diagonal, lower, upper],
            offsets=[0, -1, 1], shape=(self.nx, self.nx),
            format='csr')

        return matrix

    def step(self, u_start: VectorHeat1D2Pts, t_start: float, t_stop: float) -> VectorHeat1D2Pts:
        """
        BDF2
        """
        first, second = u_start.get_values()
        rhs = (4 / 3) * second - \
              (1 / 3) * first + \
              (2 / 3) * self.rhs(self.x, t_stop) * (t_stop - t_start - self.dt)

        tmp1 = spsolve((t_stop - t_start - self.dt) * self.space_disc + self.identity, rhs)

        rhs = (4 / 3) * tmp1 - \
              (1 / 3) * second + \
              (2 / 3) * self.rhs(self.x, t_stop + self.dt) * self.dt

        tmp2 = spsolve(self.dt * self.space_disc + self.identity, rhs)

        ret = VectorHeat1D2Pts(u_start.size)
        ret.set_values(first_time_point=tmp1, second_time_point=tmp2)

        return ret
