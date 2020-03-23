"""
Vector and application class for the 1D heat equation
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorHeat1D(Vector):
    """
    Vector class for the 1D heat equation
    """

    def __init__(self, size):
        """
        Constructor.

        :param size: number of spatial degrees of freedom
        """
        super(VectorHeat1D, self).__init__()
        self.size = size
        self.values = np.zeros(size)

    def __add__(self, other):
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        tmp = VectorHeat1D(self.size)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        tmp = VectorHeat1D(self.size)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        """
        Norm of a vector object

        :return: 2-norm of vector object
        """
        return np.linalg.norm(self.values)

    def clone_zero(self):
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """
        return VectorHeat1D(self.size)

    def clone_rand(self):
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        tmp = VectorHeat1D(self.size)
        tmp.set_values(np.random.rand(self.size))
        return tmp

    def set_values(self, values):
        """
        Set vector data

        :param values: values for vector object
        """
        self.values = values

    def get_values(self):
        """
        Get vector data

        :return: values of vector object
        """
        return self.values


class Heat1D(Application):
    """
    Application class for the heat equation in 1D space,
        u_t - a*u_xx = b(x,t),  a > 0, x in [x_start,x_end], t in [0,T],
    with homogeneous Dirichlet boundary conditions in space.
    """

    def __init__(self, x_start, x_end, nx, a, init_cond=lambda x: x * 0, rhs=lambda x, t: x * 0, *args, **kwargs):
        """
        Constructor.

        :param x_start: left interval bound of spatial domain
        :param x_end: right interval bound of spatial domain
        :param nx: number of spatial degrees of freedom
        :param a: thermal conductivity
        :param init_cond: initial condition
        :param rhs: right-hand side
        """

        super(Heat1D, self).__init__(*args, **kwargs)
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

        # Set the data structure for any user-defined time point
        self.vector_template = VectorHeat1D(self.nx)

        # Set initial condition
        self.init_cond = init_cond
        self.vector_t_start = VectorHeat1D(self.nx)
        self.vector_t_start.set_values(self.init_cond(self.x))

    def compute_matrix(self):
        """
        Define spatial discretization matrix for 1D heat equation

        Second-order central finite differences with matrix stencil
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

    def step(self, u_start: VectorHeat1D, t_start: float, t_stop: float) -> VectorHeat1D:
        """
        Time integration routine for 1D heat equation example problem:
            Backward Euler (BDF1)

        One-step method
           u_i = (I + dt*L)^{-1} * (u_{i-1} + dt*b_i),
        where L = self.space_disc is the spatial discretization operator

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        tmp = u_start.get_values()
        tmp = spsolve((t_stop - t_start) * self.space_disc + self.identity,
                      tmp + self.rhs(self.x, t_stop) * (t_stop - t_start))
        ret = VectorHeat1D(len(tmp))
        ret.set_values(tmp)
        return ret
