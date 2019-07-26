import numpy as np
from abc import ABC, abstractmethod


class Application(ABC):
    """
    """

    def __init__(self, t_start, t_stop, nt):
        """
        """
        self.t_start = t_start
        self.t_end = t_stop
        self.nt = nt
        self.t = np.linspace(self.t_start, self.t_end, nt)
        self.vector = 0
        self._u = None

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        self._u = value

    @abstractmethod
    def step(self, u_start, t_start, t_stop):
        pass
