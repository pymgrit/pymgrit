"""
Parallel Model
"""

import time

from pymgrit.core import application
from pymgrit.core import vector_standard


class ParallelModel(application.Application):
    """
    Problem for test the parallel model.
    """

    def __init__(self, sleep, *args, **kwargs):
        super(ParallelModel, self).__init__(*args, **kwargs)

        self.sleep = sleep
        self.u = vector_standard.VectorStandard(1)  # Create initial value solution
        self.count_solves = 0
        self.runtime_solves = 0

    def step(self, u_start: vector_standard.VectorStandard, t_start: float,
             t_stop: float) -> vector_standard.VectorStandard:
        """
        :param u_start:
        :param t_start:
        :param t_stop:
        :return:
        """
        start = time.time()
        time.sleep(self.sleep)
        ret = vector_standard.VectorStandard(1)
        stop = time.time()
        self.runtime_solves += stop - start
        self.count_solves += 1
        return ret
