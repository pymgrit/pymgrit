"""
Heat equation 2-d example
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from typing import Callable, Union

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
        self.values = values

    def get_values(self):
        return self.values


class Heat2D(Application):
    """
    Application class for the heat equation in 2D space
    Time discretization schemes:
      Forward Euler
      Backward Euler
      Crank-Nicolson
    """

    def __init__(self, lx: float, ly: float, nx: int, ny: int, a: float, rhs: Callable = lambda x, y, t: 0 * x * y,
                 init_con_fcn: Callable = lambda x, y: x * y * 0, method: str = 'BE',
                 u_b_0x: Union[int, float, Callable] = 0, u_b_0y: Union[int, float, Callable] = 0,
                 u_b_lx: Union[int, float, Callable] = 0, u_b_ly: Union[int, float, Callable] = 0, *args, **kwargs):
        super(Heat2D, self).__init__(*args, **kwargs)

        self.nx = nx
        self.ny = ny
        self.x = np.linspace(0, lx, self.nx)
        self.y = np.linspace(0, ly, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.a = a
        self.lx = lx
        self.ly = ly
        self.x_2d = self.x[:, np.newaxis]
        self.y_2d = self.y[np.newaxis, :]
        self.fx = (self.a / self.dx ** 2)
        self.fy = (self.a / self.dy ** 2)
        self.rhs = rhs

        if method == 'BE':
            self.theta = 1
        elif method == 'FE':
            self.theta = 0
        elif method == 'CN':
            self.theta = 0.5
        else:
            raise Exception("Unknown method. Choose BE (Backward Euler), FE (Forward Euler) or CN (Crank-Nicolson")

        # Set boundary conditions
        if isinstance(u_b_0x, (float, int)):
            self.u_b_0x: Callable = lambda t: u_b_0x
        elif callable(u_b_0x):
            self.u_b_0x: Callable = u_b_0x
        else:
            raise Exception("Choose float, int or function for boundary condition u_b_0x")

        if isinstance(u_b_0y, (float, int)):
            self.u_b_0y: Callable = lambda t: u_b_0y
        elif callable(u_b_0y):
            self.u_b_0y: Callable = u_b_0y
        else:
            raise Exception("Choose float, int or function for boundary condition u_b_0x")

        if isinstance(u_b_lx, (float, int)):
            self.u_b_lx: Callable = lambda t: u_b_lx
        elif callable(u_b_lx):
            self.u_b_lx: Callable = u_b_lx
        else:
            raise Exception("Choose float, int or function for boundary condition u_b_0x")

        if isinstance(u_b_ly, (float, int)):
            self.u_b_ly: Callable = lambda t: u_b_ly
        elif callable(u_b_ly):
            self.u_b_ly: Callable = u_b_ly
        else:
            raise Exception("Choose float, int or function for boundary condition u_b_0x")

        self.space_disc = self.compute_matrix()
        self.identity = identity(self.nx * self.ny, dtype='float', format='csr')

        self.vector_template = VectorHeat2D(self.nx, self.ny)
        self.vector_t_start = VectorHeat2D(self.nx, self.ny)  # Create initial value solution
        self.vector_t_start.set_values(init_con_fcn(self.x_2d, self.y_2d))  # Set initial value

    def compute_rhs(self, u_start: np.ndarray, t_start: float, t_stop: float):
        """
        Define right hand side corresponding to time discretization
        """
        b = np.zeros((self.nx, self.ny))

        b[:, 0] = self.u_b_0x(t_stop)
        b[:, -1] = self.u_b_lx(t_stop)
        b[0, :] = self.u_b_0y(t_stop)
        b[-1, :] = self.u_b_ly(t_stop)

        if self.theta == 1:
            b[1:-1, 1:-1] = u_start[1:-1, 1:-1] + \
                            self.theta * (t_stop - t_start) * self.rhs(x=self.x_2d[1:-1], y=self.y_2d[:, 1:-1],
                                                                       t=t_stop)
        else:
            b[1:-1, 1:-1] = u_start[1:-1, 1:-1] + \
                            (1 - self.theta) * \
                            ((self.fx * (t_stop - t_start) * (
                                    u_start[2:, 1:-1] - 2 * u_start[1:-1, 1:-1] + u_start[:-2, 1:-1])) +
                             (self.fy * (t_stop - t_start) * (
                                     u_start[1:-1, 2:] - 2 * u_start[1:-1, 1:-1] + u_start[1:-1, :-2]))) + \
                            self.theta * (t_stop - t_start) * self.rhs(x=self.x_2d[1:-1], y=self.y_2d[:, 1:-1],
                                                                       t=t_stop) + \
                            (1 - self.theta) * (t_stop - t_start) * self.rhs(x=self.x_2d[1:-1], y=self.y_2d[:, 1:-1],
                                                                             t=t_start)

        return b.T.flatten()

    def compute_matrix(self):
        """
        Define spatial discretization matrix for 2D heat equation
        Second-order central finite differences in space.
        """
        n = self.nx * self.ny

        main = np.ones(n) * (2 * self.theta * (self.fx + self.fy))
        main[:self.nx] = main[-self.nx:] = 0
        main[self.nx::self.nx] = main[2 * self.nx - 1::self.nx] = 0

        upper = np.ones(n - 1) * -self.theta * self.fx
        upper[:self.nx] = upper[-self.nx:] = 0
        upper[self.nx::self.nx] = upper[2 * self.nx - 1:: self.nx] = 0

        lower = np.ones(n - 1) * -self.theta * self.fx
        lower[:self.nx - 1] = lower[-self.nx:] = 0
        lower[self.nx - 1::self.nx] = lower[2 * self.nx - 2:: self.nx] = 0

        upper2 = np.ones(n - self.nx) * -self.theta * self.fy
        upper2[:self.nx] = 0
        upper2[self.nx::self.nx] = upper2[2 * self.nx - 1:: self.nx] = 0

        lower2 = np.ones(n - self.nx) * -self.theta * self.fy
        lower2[-self.nx:] = 0
        lower2[::self.nx] = lower2[self.nx - 1::self.nx] = 0

        matrix = diags(diagonals=[main, lower, upper, lower2, upper2],
                       offsets=[0, -1, 1, -self.nx, self.nx],
                       shape=(n, n), format='csr')
        return matrix

    def step(self, u_start: VectorHeat2D, t_start: float, t_stop: float) -> VectorHeat2D:
        """
        Time integration routine for 2D heat equation example problem:
            Backward Euler in time
            Forward Euler in time
            Crank-Nicolson in time

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        rhs = self.compute_rhs(u_start=u_start.get_values(), t_start=t_start, t_stop=t_stop)
        new = spsolve((t_stop - t_start) * self.space_disc + self.identity, rhs)
        ret = VectorHeat2D(self.nx, self.ny)
        ret.set_values(new.reshape(self.ny, self.nx).T)
        return ret
