"""
Parallel Model
"""

import time
import numpy as np

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorParallelModel(Vector):
    """
    Vector for the parallel model
    """

    def __init__(self, size):
        super(VectorParallelModel, self).__init__()
        self.size = size
        self.values = np.zeros(size)

    def __add__(self, other):
        tmp = VectorParallelModel(self.size)
        tmp.set_values(self.get_values() + other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorParallelModel(self.size)
        tmp.set_values(self.get_values() - other.get_values())
        return tmp

    def norm(self):
        return np.linalg.norm(self.values)

    def clone_zero(self):
        return VectorParallelModel(self.size)

    def clone_rand(self):
        tmp = VectorParallelModel(self.size)
        tmp.set_values(np.random.rand(self.size))
        return tmp

    def set_values(self, values):
        self.values = values

    def get_values(self):
        return self.values


class ParallelModel(Application):
    """
    Problem for test the parallel model.
    """

    def __init__(self, sleep, *args, **kwargs):
        super(ParallelModel, self).__init__(*args, **kwargs)

        self.sleep = sleep
        self.vector_template = VectorParallelModel(1)  # Create initial value solution
        self.vector_t_start = VectorParallelModel(1)  # Create initial value solution
        self.count_solves = 0
        self.runtime_solves = 0

    def step(self, u_start: VectorParallelModel, t_start: float, t_stop: float) -> VectorParallelModel:
        """
        :param u_start:
        :param t_start:
        :param t_stop:
        :return:
        """
        start = time.time()
        time.sleep(self.sleep)
        ret = VectorParallelModel(1)
        stop = time.time()
        self.runtime_solves += stop - start
        self.count_solves += 1
        return ret
