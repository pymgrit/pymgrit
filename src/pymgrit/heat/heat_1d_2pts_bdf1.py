"""
Application class for 1D heat problem using BDF1 time integration

Note: values at two consecutive time points are grouped as pairs
"""

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
    with homogeneous Dirichlet boundary conditions in space
    """

    def __init__(self, x_start: float, x_end: float, nx: int, dtau: float, a: float,
                 init_cond: Callable = lambda x: x * 0, rhs: Callable = lambda x, t: x * 0, *args,
                 **kwargs):
        """
        Constructor.

        :param x_start: left interval bound of spatial domain
        :param x_end: right interval bound of spatial domain
        :param nx: number of spatial degrees of freedom
        :param dtau: time-step size within pair
        :param a: thermal conductivity
        :param init_cond: initial condition
        :param rhs: right-hand side
        """

        super(Heat1DBDF1, self).__init__(*args, **kwargs)
        # Spatial domain with homogeneous Dirichlet boundary conditions
        self.x_start = x_start
        self.x_end = x_end
        self.x = np.linspace(self.x_start, self.x_end, nx)
        self.x = self.x[1:-1]
        self.nx = nx - 2
        self.dx = self.x[1] - self.x[0]

        # Thermal conductivity
        self.a = a

        # Set (spatial) identity matrix and spatial discretization matrix
        self.identity = identity(self.nx, dtype='float', format='csr')
        self.space_disc = self.compute_matrix()

        # Set right-hand side routine
        self.rhs = rhs

        # Set the data structure for any user-defined time-point pairs
        self.vector_template = VectorHeat1D2Pts(self.nx, dtau)

        # Set initial condition
        self.init_cond = init_cond
        self.vector_t_start = VectorHeat1D2Pts(self.nx, dtau)
        tmp1 = self.init_cond(self.x)
        # Take one BDF1 step to get value at time dtau
        tmp2 = spsolve(dtau * self.space_disc + self.identity,
                       tmp1 + self.rhs(self.x, self.t[0] + dtau) * dtau)
        self.vector_t_start.set_values(first_time_point=tmp1, second_time_point=tmp2, dtau=dtau)

    def compute_matrix(self):
        """
        Define spatial discretization matrix for heat equation problem.

        Discretization is centered finite differences with matrix stencil
           (a / dx^2) * [-1  2  -1]
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
            Backward Euler (BDF1)

        One-step method
           u_i = (I + dt*L)^{-1} * (u_{i-1} + dt*b_i),
        where L = self.space_disc is the spatial discretization operator

        Note: step takes two BDF1 steps
          * one step from (t_start + dtau) to t_stop
          * one step from t_stop to (t_stop + dtau)

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        first, second, dtau = u_start.get_values()
        tmp1 = spsolve((t_stop - t_start - dtau) * self.space_disc + self.identity,
                       second + self.rhs(self.x, t_stop) * (t_stop - t_start - dtau))

        tmp2 = spsolve(dtau * self.space_disc + self.identity,
                       tmp1 + self.rhs(self.x, t_stop + dtau) * dtau)

        ret = VectorHeat1D2Pts(u_start.size, u_start.dtau)
        ret.set_values(first_time_point=tmp1, second_time_point=tmp2, dtau=dtau)
        return ret
