"""
Local convergence tests, where each time point converges in a different iteration.
"""
import numpy as np

from pymgrit.core.application import Application
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.vector import Vector


class SimpleVector(Vector):
    """
    Vector class for the Dahlquist test equation
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __add__(self, other):
        if self.value == 0 or other.get_values() == 0:
            tmp = SimpleVector(0)
        else:
            tmp = SimpleVector(100)
        return tmp

    def __sub__(self, other):
        if self.value == 0 or other.get_values() == 0:
            tmp = SimpleVector(0)
        else:
            tmp = SimpleVector(100)
        return tmp

    def __mul__(self, other):
        if self.value == 0:
            tmp = SimpleVector(0)
        else:
            tmp = SimpleVector(100)
        return tmp

    def norm(self):
        return 0 if self.value == 0 else 100

    def clone(self):
        return SimpleVector(self.value)

    def clone_zero(self):
        return SimpleVector(self.value)

    def clone_rand(self):
        return SimpleVector(self.value)

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value

    def pack(self):
        return self.value

    def unpack(self, value):
        self.value = value


class SimpleApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_template = SimpleVector(1)  # Set the class to be used for each time point
        self.vector_t_start = SimpleVector(0)  # Set the initial condition
        self.counter = np.zeros(self.nt, dtype=int)

    def step(self, u_start: SimpleVector, t_start: float, t_stop: float) -> SimpleVector:
        int_t_stop = int(t_stop)
        if self.counter[int_t_stop] >= int(t_stop)*2:
            tmp = SimpleVector(0)
        else:
            tmp = SimpleVector(100)
        self.counter[int_t_stop]+=1
        return tmp


def main():
    level_0 = SimpleApp(t_start=0, t_stop=10, nt=11)
    level_1 = SimpleApp(t_interval=level_0.t)
    mgrit = Mgrit(problem=[level_0, level_1], tol=1e-10,
                  nested_iteration=False, max_iter=15, cf_iter=0, conv_crit=3)
    info = mgrit.solve()
    print(mgrit.comm_time_rank, info['conv'])


if __name__ == '__main__':
    main()
