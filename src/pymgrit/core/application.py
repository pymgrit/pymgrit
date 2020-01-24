"""
Abstract class for problems. Each problem has to be a child
"""
from abc import ABC, abstractmethod
import numpy as np


class Application(ABC):
    """
    Abstract class for problems. Each problem has to be a child
    """

    def __init__(self, t_start: float = None, t_stop: float = None, nt: int = None,
                 t_interval: np.ndarray = None) -> None:
        """
        Initiate the time information
        :param t_start: Start point
        :param t_stop: End point
        :param nt: Number of time points
        """

        if t_interval is None:
            if t_start is None or t_stop is None or nt is None:
                raise Exception('Specify an interval by t_start, t_stop and nt or by t_interval')
            self.t_start = t_start
            self.t_end = t_stop
            self.nt = nt
            self.t = np.linspace(self.t_start, self.t_end, nt)
            self._u = None
        else:
            if not isinstance(t_interval, np.ndarray):
                raise Exception('t_interval has the wrong type. Should be a numpy array')
            self.t_start = t_interval[0]
            self.t_end = t_interval[-1]
            self.nt = len(t_interval)
            self.t = t_interval
            self._u = None

    @property
    def u(self):
        """
        Property u
        :return:
        """
        return self._u

    @u.setter
    def u(self, value):
        """
        Property u
        :return:
        """
        self._u = value

    @abstractmethod
    def step(self, u_start: object, t_start: float, t_stop: float) -> object:
        """
        Time stepper for problem
        :param u_start: Start solution
        :param t_start: Start time point
        :param t_stop: End time point
        """
