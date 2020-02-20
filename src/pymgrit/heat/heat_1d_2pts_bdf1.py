import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

from pymgrit.core.application import Application
from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts
from scipy.sparse import identity


class Heat1DBDF1(Application):
    """
    Heat equation 1-d example
    u_t - a*u_xx = b(x,t),  a > 0, x in [0,1], t in [0,T]
         u(0,t)  = u(1,t) = 0,   t in [0,T]
         u(x,0)  = sin(pi*x),    x in [0,1]
    with RHS b(x,t) = -sin(pi*x)(sin(t) - a*pi^2*cos(t))
    => solution u(x,t) = sin(pi*x)*cos(t)
    """

    def __init__(self, x_start, x_end, nx, dt, a, *args, **kwargs):
        super(Heat1DBDF1, self).__init__(*args, **kwargs)
        self.x_start = x_start  # lower interval bound of spatial domain
        self.x_end = x_end  # upper interval bound of spatial domain
        self.x = np.linspace(self.x_start, self.x_end, nx)  # Spatial domain
        self.x = self.x[1:-1]  # homogeneous BCs
        self.nx = nx - 2  # homogeneous BCs
        self.dt = dt  # time-step size
        self.a = a  # diffusion coefficient
        self.dx = self.x[1] - self.x[0]

        self.u_ex = self.u_exact_complete(x=self.x, t=np.linspace(self.t_start, self.t_end, (self.nt - 1) * 2 + 1))

        self.identity = identity(self.nx, dtype='float', format='csr')

        self.space_disc = self.compute_matrix()

        self.vector_template = VectorHeat1D2Pts(self.nx)  # Create initial value solution
        self.vector_t_start = VectorHeat1D2Pts(self.nx)

        self.vector_t_start.set_values(first_time_point=self.u_exact(self.x, self.t[0]),
                                       second_time_point=spsolve(self.dt * self.space_disc + self.identity,
                                                                 self.u_exact(self.x, self.t[0]) +
                                                                 self.f(self.x, self.t[0] + dt) *
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

    @staticmethod
    def heat_sparse(nx, fac):
        """
        Central FD in space
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

    def u_exact(self, x, t):
        """
        Solution for one time point
        """
        return np.sin(np.pi * x) * np.cos(t)

    def f(self, x, t):
        """
        Right-hand-side
        """
        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def u_exact_complete(self, x, t):
        """
        Solution for all time points
        """
        ret = np.zeros((np.size(t), np.size(x)))
        for i in range(np.size(t)):
            ret[i] = self.u_exact(x, t[i])
        return ret

    def step(self, u_start: VectorHeat1D2Pts, t_start: float, t_stop: float) -> VectorHeat1D2Pts:
        """
        Backward Euler in time
        At each time step i = 1, ..., nt+1, we obtain the linear system
        | 1+2ar   -ar                     | |  u_{1,i}   |   |  u_{1,i-1}   |
        |  -ar   1+2ar  -ar               | |  u_{2,i}   |   |  u_{2,i-1}   |
        |         ...   ...    ...        | |    ...     | = |     ...      |
        |               -ar   1+2ar  -ar  | | u_{nx-1,i} |   | u_{nx-1,i-1} |
        |                      -ar  1+2ar | |  u_{nx,i}  |   |  u_{nx,i-1}  |

                                                             |  dt*b_{1,i}   |
                                                             |  dt*b_{2,i}   |
                                                             |     ...       |
                                                             | dt*b_{nx-1,i} |
                                                             |  dt*b_{nx,i}  |
        with r = (dt/dx^2), which we denote by
        Mu_i = u_{i-1} + dt*b_i.
        This leads to the time-stepping problem u_i = M^{-1}(u_{i-1} + dt*b_i)
        which is implemented as time integrator function Phi u_i = Phi(u_{i-1}, t_{i}, t_{i-1}, app)
        """
        first, second = u_start.get_values()
        tmp1 = spsolve((t_stop - t_start - self.dt) * self.space_disc + self.identity,
                       second + self.f(self.x, t_stop) * (t_stop - t_start - self.dt))

        tmp2 = spsolve(self.dt * self.space_disc + self.identity,
                       tmp1 + self.f(self.x, t_stop + self.dt) * self.dt)

        ret = VectorHeat1D2Pts(u_start.size)
        ret.set_values(first_time_point=tmp1, second_time_point=tmp2)
        return ret
