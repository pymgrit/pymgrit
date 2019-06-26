import numpy as np
from abc import ABC, abstractmethod


class Application(ABC):
    """
    """

    def __init__(self, nx=0, t_start=0, t_stop=0, nt=0):
        """
        """
        self.nx = nx
        self.t_start = t_start
        self.t_end = t_stop
        self.nt = nt
        self.t = np.linspace(self.t_start, self.t_end, nt)
        self.spatial_coarsening_max_lvl = 0

    @abstractmethod
    def setup(self, lvl_max, t, spatial_coarsening):
        pass

    @abstractmethod
    def initial_value(self):
        pass

    @abstractmethod
    def phi(self, u_start, t_start, t_stop, app):
        pass

    @abstractmethod
    def restriction(self, u, app=None):
        pass

    @abstractmethod
    def interpolation(self, u, app=None):
        pass

    @abstractmethod
    def info(self):
        pass
