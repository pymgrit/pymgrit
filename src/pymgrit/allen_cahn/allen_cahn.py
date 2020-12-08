"""
Vector and application class for the 2D Allen-Cahn equation

Based on https://epubs.siam.org/doi/abs/10.1137/080738398?mobileUi=0 and
https://dl.acm.org/doi/10.1145/3310410
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymgrit.core.vector import Vector
from pymgrit.core.application import Application


class VectorAllenCahn2D(Vector):
    """
    Vector class for the 2D Allen-Cahn equation
    """

    def __init__(self, nx, ny):
        """
        Constructor.

        :param nx: number of degrees of freedom in x-direction
        :param ny: number of degrees of freedom in y-direction
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.values = np.zeros((self.nx, self.ny))

    def __add__(self, other):
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        tmp = VectorAllenCahn2D(self.nx, self.ny)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        tmp = VectorAllenCahn2D(self.nx, self.ny)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def __mul__(self, other):
        """
        Multiplication of a vector object and a float (self and other)

        :param other: object to be multiplied with self
        :return: difference of vector object self and input object other
        """
        tmp = VectorAllenCahn2D(self.nx, self.ny)
        tmp.set_values(self.get_values() * other)
        return tmp

    def norm(self):
        """
        Norm of a vector object

        :return: 2-norm of vector object
        """
        return np.linalg.norm(self.values)

    def clone(self):
        """
        Clone vector object

        :rtype: vector object with zero values
        """
        tmp = VectorAllenCahn2D(self.nx, self.ny)
        tmp.set_values(self.get_values())
        return tmp

    def clone_zero(self):
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """
        return VectorAllenCahn2D(self.nx, self.ny)

    def clone_rand(self):
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        tmp = VectorAllenCahn2D(self.nx, self.ny)
        tmp.set_values(np.random.rand(self.nx, self.ny))
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

    def pack(self):
        """
        Pack data

        :return: values of vector object
        """
        return self.values

    def unpack(self, values):
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.values = values


class AllenCahn(Application):
    """
    Application class for the Allen-Cahn in 2D space,
        u_t =  (u_xx + u_yy) + 1/ eps^2 u(1-u),  a > 0,
        on [-0.5, 0.5] x [-0.5, 0.5] x (t_start, t_end],
    with periodic boundary conditions, eps > 0,
    and subject to the initial condition
        u(x,0)  = tanh((R0-|x|)/sqrt(2eps).
    """

    def __init__(self, nx=128, nu=2, eps=0.04, newton_maxiter=100, newton_tol=1e-12, lin_tol=1e-12, lin_maxiter=100,
                 radius=0.25, method='IMPL', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nu = nu
        self.eps = eps
        self.newton_maxiter = newton_maxiter
        self.newton_tol = newton_tol
        self.lin_tol = lin_tol
        self.lin_maxiter = lin_maxiter
        self.radius = radius
        self.nx = nx
        self.ny = nx
        self.method = method

        if self.method == 'IMPL' or self.method == 'IMEX' or self.method == 'CN':
            pass
        else:
            raise Exception("Unknown method. Choose IMPL (implicit), IMEX (implicit-explicit) or CN (Crank-Nicolson")

        self.dx = 1.0 / self.nx
        self.space_disc = self.compute_matrix()
        self.id = sp.eye(self.nx * self.ny)
        self.x = np.linspace(start=-0.5, stop=0.5, num=self.nx)

        self.vector_t_start = self.initial_guess()
        self.vector_template = VectorAllenCahn2D(nx=self.nx, ny=self.ny)

    def compute_matrix(self):
        """
        Define spatial discretization matrix

        :return: space discretizaton
        """
        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([self.nx - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - self.nx))

        space_disc = sp.diags(dstencil, doffsets, shape=(self.nx, self.ny), format='csc')
        space_disc = sp.kron(space_disc, sp.eye(self.nx)) + sp.kron(sp.eye(self.ny), space_disc)
        space_disc *= 1.0 / (self.dx ** 2)

        return space_disc

    def step(self, u_start: VectorAllenCahn2D, t_start: float, t_stop: float) -> VectorAllenCahn2D:
        """
        Time integration routine for 2D Allen-Cahn problem:

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        ret = VectorAllenCahn2D(self.nx, self.ny)
        if self.method == 'IMEX':
            new = u_start.get_values().flatten()
            rhs = new + (t_stop - t_start) * (1 / self.eps ** 2 * new * (1.0 - new ** self.nu))
            new = spsolve(self.id - (t_stop - t_start) * self.space_disc, rhs)
            ret.set_values(new.reshape(self.nx, self.ny))
        else:
            new = u_start.get_values().flatten()
            old = u_start.get_values().flatten()
            eps2 = self.eps ** 2

            if self.method == 'CN':
                fac = (t_stop - t_start) / 2
                rhs = old + (t_stop - t_start) / 2 * (
                        self.space_disc.dot(old) + 1.0 / eps2 * old * (1.0 - old ** self.nu))
            else:
                fac = (t_stop - t_start)
                rhs = old

            n = 0
            while 0 < self.newton_maxiter:
                g = new - fac * (self.space_disc.dot(new) + 1.0 / eps2 * new * (1.0 - new ** self.nu)) - rhs
                if np.linalg.norm(g, np.inf) < self.newton_tol:
                    break
                dg = self.id - fac * (
                        self.space_disc + 1.0 / eps2 * sp.diags((1.0 - (self.nu + 1) * new ** self.nu), offsets=0))
                new -= spsolve(dg, g)
                n += 1
            ret.set_values(new.reshape(self.nx, self.ny))
        return ret

    def initial_guess(self):
        """
        Set initial condition

        :return: Initial guess
        """
        initial = VectorAllenCahn2D(nx=self.nx, ny=self.ny)
        values = initial.get_values()
        for i in range(self.nx):
            for j in range(self.ny):
                r2 = self.x[i] ** 2 + self.x[j] ** 2
                values[i, j] = np.tanh((self.radius - np.sqrt(r2)) / (np.sqrt(2) * self.eps))
        initial.set_values(values=values)
        return initial

    def exact_radius(self, t):
        """
        Compute exact radius

        :return: Exact radius
        """
        return np.sqrt(max(self.radius ** 2 - 2.0 * t, 0))

    def compute_radius(self, u):
        """
        Compute radius of u

        :return: Radius
        """
        return np.sqrt(np.count_nonzero(u.get_values() >= 0.0) / np.pi) * self.dx
