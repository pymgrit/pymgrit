import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from typing import Callable

from pymgrit.core.application import Application
from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts


class Heat1DBDF1(Application):
    """
    Application class for the heat equation in 1D space,
        u_t - a*u_xx = b(x,t),  a > 0, x in [x_start,x_end], t in [0,T],
    """

    def __init__(self, x_start: float, x_end: float, nx: int, dt: float, a: float,
                 init_con_fnc: Callable = lambda x: x * 0, rhs: Callable = lambda x, t: x * 0, *args,
                 **kwargs):
        super(Heat1DBDF1, self).__init__(*args, **kwargs)
        self.x_start = x_start  # lower interval bound of spatial domain
        self.x_end = x_end  # upper interval bound of spatial domain
        self.x = np.linspace(self.x_start, self.x_end, nx)  # Spatial domain
        self.x = self.x[1:-1]  # homogeneous BCs
        self.nx = nx - 2  # homogeneous BCs
        self.dt = dt  # time-step size
        self.a = a  # diffusion coefficient
        self.dx = self.x[1] - self.x[0]
        self.identity = identity(self.nx, dtype='float', format='csr')
        self.init_con_fnc = init_con_fnc
        self.rhs = rhs

        # set spatial discretization matrix
        self.space_disc = self.compute_matrix()

        self.vector_template = VectorHeat1D2Pts(self.nx)  # Create initial value solution
        self.vector_t_start = VectorHeat1D2Pts(self.nx)

        self.vector_t_start.set_values(first_time_point=self.init_con_fnc(self.x, self.t[0]),
                                       second_time_point=spsolve(self.dt * self.space_disc + self.identity,
                                                                 self.init_con_fnc(self.x, self.t[0]) +
                                                                 self.rhs(self.x, self.t[0] + self.dt) *
                                                                 self.dt))

    def compute_matrix(self):
        """
        Space discretization
        """

        fac = self.a / self.dx ** 2

        diagonal = np.ones(self.nx) * 2 * fac
        lower = np.ones(self.nx - 1) * -fac
        upper = np.ones(self.nx - 1) * -fac

        matrix = sp.diags(
            diagonals=[diagonal, lower, upper],
            offsets=[0, -1, 1], shape=(self.nx, self.nx),
            format='csr')

        return matrix

    def step(self, u_start: VectorHeat1D2Pts, t_start: float, t_stop: float) -> VectorHeat1D2Pts:
        """
        Time integration routine for 1D heat equation example problem:
            Backward Euler in time

        At each time step i = 1, ..., nt+1, we obtain the linear system
        | 1+2ar   -ar                     | |  u_{1,i}   |   |  u_{1,i-1}   |   |  dt*b_{1,i}   |
        |  -ar   1+2ar  -ar               | |  u_{2,i}   |   |  u_{2,i-1}   |   |  dt*b_{2,i}   |
        |         ...   ...    ...        | |    ...     | = |     ...      | + |     ...       |
        |               -ar   1+2ar  -ar  | | u_{nx-1,i} |   | u_{nx-1,i-1} |   | dt*b_{nx-1,i} |
        |                      -ar  1+2ar | |  u_{nx,i}  |   |  u_{nx,i-1}  |   |  dt*b_{nx,i}  |

        with r = (dt/dx^2), which we denote by
            Mu_i = u_{i-1} + dt*b_i.
        This leads to the time-stepping problem u_i = M^{-1}(u_{i-1} + dt*b_i)
        which is implemented as time integrator function Phi u_i = Phi(u_{i-1}, t_{i}, t_{i-1}, app)

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        first, second = u_start.get_values()
        tmp1 = spsolve((t_stop - t_start - self.dt) * self.space_disc + self.identity,
                       second + self.rhs(self.x, t_stop) * (t_stop - t_start - self.dt))

        tmp2 = spsolve(self.dt * self.space_disc + self.identity,
                       tmp1 + self.rhs(self.x, t_stop + self.dt) * self.dt)

        ret = VectorHeat1D2Pts(u_start.size)
        ret.set_values(first_time_point=tmp1, second_time_point=tmp2)
        return ret
