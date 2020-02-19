"""
Lorentz equation
"""

import numpy as np

from scipy.integrate import solve_ivp

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorLorentz(Vector):
    """
    Vector for the lorentz equation
    """

    def __init__(self):
        super(VectorLorentz, self).__init__()
        self.value = np.zeros(3)

    def __add__(self, other):
        tmp = VectorLorentz()
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorLorentz()
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.value)

    def clone_zero(self):
        return VectorLorentz()

    def clone_rand(self):
        tmp = VectorLorentz()
        tmp.set_values(np.random.rand(3))
        return tmp

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value


def lorentz(t, y):
    sigma = 28
    r = 10
    b = 8 / 3
    return np.array([r * (y[1] - y[0]), y[0] * (sigma - y[2]) - y[1], y[0] * y[1] - b * y[2]])


class Lorentz(Application):
    """
    """

    def __init__(self, *args, **kwargs):
        super(Lorentz, self).__init__(*args, **kwargs)
        self.vector_template = VectorLorentz()  # Setting the class which is used for each time point
        self.vector_t_start = VectorLorentz()
        self.vector_t_start.set_values(np.array([20.0, 5.0, -5.0]))

    def step(self, u_start: VectorLorentz, t_start: float, t_stop: float) -> VectorLorentz:
        res = solve_ivp(fun=lorentz, y0=u_start.get_values(),
                        t_span=np.array([t_start, t_stop]),
                        t_eval=np.array([t_start, t_stop]), method='RK45')
        ret = VectorLorentz()
        ret.set_values(res.y[:, -1])
        return ret
