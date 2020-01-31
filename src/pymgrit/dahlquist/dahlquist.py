"""
Scalar ODE example
"""

import numpy as np

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorDahlquist(Vector):
    """
    Vector for the 1D advection equation
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
    Solves  u' = lambda u,
    with lambda=-1 and y(0) = 1
    """

    def __init__(self, *args, **kwargs):
        super(Dahlquist, self).__init__(*args, **kwargs)
        self.vector_template = VectorDahlquist(0)  # Create initial value solution
        self.vector_t_start = VectorDahlquist(1)  # Create initial value solution

    def step(self, u_start: VectorDahlquist, t_start: float, t_stop: float) -> VectorDahlquist:
        tmp = 1 / (1 + t_stop - t_start) * u_start.get_values()
        ret = VectorDahlquist(tmp)
        return ret
