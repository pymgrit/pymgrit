"""
Vector and application class for the Arenstorf orbit problem
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorArenstorfOrbit(Vector):
    """
    Vector for the Arenstorf orbit problem
    """

    def __init__(self):
        super(VectorArenstorfOrbit, self).__init__()
        self.y0 = 0
        self.y1 = 0
        self.y2 = 0
        self.y3 = 0

    def __add__(self, other):
        tmp = VectorArenstorfOrbit()
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorArenstorfOrbit()
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(np.array([self.y0, self.y1, self.y2, self.y3]))

    def clone_zero(self):
        return VectorArenstorfOrbit()

    def clone_rand(self):
        tmp = VectorArenstorfOrbit()
        tmp.set_values(np.random.rand(4))
        return tmp

    def set_values(self, values):
        self.y0 = values[0]
        self.y1 = values[1]
        self.y2 = values[2]
        self.y3 = values[3]

    def get_values(self):
        return np.array([self.y0, self.y1, self.y2, self.y3])

    def plot(self):
        plt.plot(self.y0, self.y1, color='red', marker='.')


def arenstorf(t, y, a, b):
    d1 = ((y[0] + a) ** 2 + y[1] ** 2) ** (3 / 2)
    d2 = ((y[0] - b) ** 2 + y[1] ** 2) ** (3 / 2)
    yp = np.array([
        y[2],
        y[3],
        y[0] + 2 * y[3] - b * (y[0] + a) / d1 - a * (y[0] - b) / d2,
        y[1] - 2 * y[2] - b * y[1] / d1 - a * y[1] / d2
    ], dtype=float)
    return yp


class ArenstorfOrbit(Application):
    """
    Application for Arenstorf orbit problem,
       x'' = x + 2y' - b*(x + a)/D_1 - a*(x - b)/D_2,
       y'' = y - 2x' - b*y/D_1 - a*y/D_2
    with a = 0.012277471, b = 1 - a,
        D_1 = ((x + a)^2 + y^2)^(3/2),  D_2 = ((x - a)^2 + y^2)^(3/2)
    and ICs
       x(0) = 0.994,  x'(0) = 0,  y(0) = 0,  y'(0) = -2.00158510637908
    """

    def __init__(self, *args, **kwargs):
        super(ArenstorfOrbit, self).__init__(*args, **kwargs)
        self.vector_template = VectorArenstorfOrbit()  # Setting the class which is used for each time point
        self.vector_t_start = VectorArenstorfOrbit()
        self.vector_t_start.set_values(np.array([0.994, 0.0, 0.0, -2.00158510637908]))

        self.a = 0.012277471
        self.b = 1 - self.a

    def step(self, u_start: VectorArenstorfOrbit, t_start: float, t_stop: float) -> VectorArenstorfOrbit:
        res = solve_ivp(fun=arenstorf, y0=u_start.get_values(), args=(self.a, self.b),
                        t_span=np.array([t_start, t_stop]),
                        t_eval=np.array([t_start, t_stop]), method='RK45')
        ret = VectorArenstorfOrbit()
        ret.set_values(res.y[:, -1])
        return ret
