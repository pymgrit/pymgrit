"""
Scalar ODE example:
Vector and application class for Dahlquist's test problem
"""

import numpy as np

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorDahlquist(Vector):
    """
    Vector class for the Dahlquist test equation
    """

    def __init__(self, value):
        super(VectorDahlquist, self).__init__()
        self.value = value

    def __add__(self, other):
        tmp = VectorDahlquist(0)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorDahlquist(0)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.value)

    def clone_zero(self):
        return VectorDahlquist(0)

    def clone_rand(self):
        tmp = VectorDahlquist(0)
        tmp.set_values(np.random.rand(1)[0])
        return tmp

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value


class Dahlquist(Application):
    """
    Application class for Dahlquist's test problem,
        u' = lambda u,
    with lambda = -1 and IC u(0) = 1
    """

    def __init__(self, method='BE', *args, **kwargs):
        """
        Initialize Dahlquist application object
        :param method: method for solving Dahlquist's equation:
                       'BE' -> Backward Euler (default)
                       'FE' -> Forwared Euler
                       'TR' -> Trapezoidal rule
                       'MR' -> Implicit Mid-point Rule
        """
        super(Dahlquist, self).__init__(*args, **kwargs)
        self.vector_template = VectorDahlquist(0)  # Set the class to be used for each time point
        self.vector_t_start = VectorDahlquist(1)  # Set the initial condition
        if method == 'BE' or method == 'FE' or method == 'TR' or method == 'MR':
            self.method = method
        else:
            raise Exception(
                'Unknown method. Choose BE (Backward Euler), FE (Forward Euler), TR (Trapezoidal rule) ' +
                'or MR (implicit mid-point rule)')

    def step(self, u_start: VectorDahlquist, t_start: float, t_stop: float) -> VectorDahlquist:
        """
        Time integration routine for Dahlquist's test problem:
            BE: Backward Euler
            FE: Forward Euler
            TR: Trapezoidal rule
            MR: implicit Mid-point Rule

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution for the input time t_stop
        """
        z = (t_stop - t_start) * -1  # Note: lambda = -1
        if self.method == 'BE':
            tmp = 1 / (1 - z) * u_start.get_values()
        elif self.method == 'FE':
            tmp = (1 + z) * u_start.get_values()
        elif self.method == 'TR':
            tmp = (1 + z / 2) / (1 - z / 2) * u_start.get_values()
        elif self.method == 'MR':
            k1 = -1 / (1 - z / 2) * u_start.get_values()
            tmp = u_start.get_values() + (t_stop - t_start) * k1
        return VectorDahlquist(tmp)
