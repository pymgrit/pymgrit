"""
Vector and application class for the 1D Brusselator system
"""

import numpy as np
import matplotlib.pyplot as plt

from pymgrit.core.vector import Vector
from pymgrit.core.application import Application


class VectorBrusselator(Vector):
    """
    Vector class for Brusselator system
    """

    def __init__(self):
        super(VectorBrusselator, self).__init__()
        self.value = np.zeros(2)

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value

    def clone_zero(self):
        return VectorBrusselator()

    def clone_rand(self):
        tmp = VectorBrusselator()
        tmp.set_values(np.random.rand(2))
        return tmp

    def __add__(self, other):
        tmp = VectorBrusselator()
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorBrusselator()
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.value)

    def plot_solution(self):
        plt.plot(self.value[0], self.value[1], color='red', marker='o')


def brusselator(t, y):
    """
    Right-hand side of Brusselator system
    :param t: time associated with the input approximate solution y
    :param y: approximate solution for the input time t
    :return: ODE right-hand side of Brusselator system
    """
    a = 1
    b = 3
    f = np.array([
        a + (y[0] ** 2) * y[1] - (b + 1) * y[0],
        b * y[0] - (y[0] ** 2) * y[1]
    ], dtype=float)
    return f


class Brusselator(Application):
    """
    Application class for Brusselator system,
       x' = A + x^2y - (B + 1)x,
       y' = Bx - x^2y,
    with A = 1, B = 3, and ICs
       x(0) = 0,  y(0) = 1
    """

    def __init__(self, *args, **kwargs):
        super(Brusselator, self).__init__(*args, **kwargs)
        self.vector_template = VectorBrusselator()
        self.vector_t_start = VectorBrusselator()

        # set initial condition
        self.vector_t_start.set_values(np.array([0.0, 1.0]))

        # set parameters (concentrations) of the problem
        self.a = 1
        self.b = 3

    def step(self, u_start: VectorBrusselator, t_start: float, t_stop: float) -> VectorBrusselator:
        """
        Time integration routine for Brusselator system: RK4

           0   |
         1 / 2 | 1 / 2
         1 / 2 |   0    1 / 2
           1   |   0      0      1
         ------+----------------------------
               | 1 / 6  1 / 3  1 / 3  1 / 6

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution for the input time t_stop
        """

        dt = t_stop - t_start

        k1 = brusselator(t_start, u_start.get_values())
        k2 = brusselator(t_start + dt / 2, u_start.get_values() + dt / 2 * k1)
        k3 = brusselator(t_start + dt / 2, u_start.get_values() + dt / 2 * k2)
        k4 = brusselator(t_start + dt, u_start.get_values() + dt * k3)

        u = u_start.get_values() + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u_stop = VectorBrusselator()
        u_stop.set_values(u)
        return u_stop
