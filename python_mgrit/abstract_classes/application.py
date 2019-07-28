import numpy as np
from abc import ABC, abstractmethod


class Application(ABC):
    """
    Abstract class for problems. Each problem has to be a child
    """

    def __init__(self, t_start: float, t_stop: float, nt: int) -> None:
        """
        Initiate the time information
        :param t_start: Start point
        :param t_stop: End point
        :param nt: Number of time points
        """
        self.t_start = t_start
        self.t_end = t_stop
        self.nt = nt
        self.t = np.linspace(self.t_start, self.t_end, nt)
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
