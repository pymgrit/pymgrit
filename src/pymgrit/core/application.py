"""
Abstract application class.
Every application must inherit from this class.
Required attributes:
  - vector_template
  - vector_t_start
Required functions:
  - step
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from pymgrit.core.vector import Vector


class MetaApplication(ABCMeta):
    """
    MetaClass for the application class.
    Checks whether required attributes are set.
    """
    required_attributes = []

    def __call__(cls, *args, **kwargs):
        obj = super(MetaApplication, cls).__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not hasattr(obj, attr_name):
                raise ValueError('required attribute (%s) not set' % attr_name)
        return obj


class Application(object, metaclass=MetaApplication):
    """
    Abstract application class
    Required attributes:
      - vector_template
      - vector_t_start
    Required functions:
      - step
    """
    required_attributes = ['vector_template', 'vector_t_start']

    def __init__(self, t_start: float = None, t_stop: float = None, nt: int = None,
                 t_interval: np.ndarray = None) -> None:
        """
        Initialize the time information
        :param t_start: Start point
        :param t_stop: End point
        :param nt: Number of time points.
        """

        if t_interval is None:
            if t_start is None or t_stop is None or nt is None:
                raise Exception('Specify an interval by t_start, t_stop and nt or by t_interval')
            self.t_start = t_start
            self.t_end = t_stop
            self.nt = nt
            self.t = np.linspace(self.t_start, self.t_end, nt)
        else:
            if not isinstance(t_interval, np.ndarray):
                raise Exception('t_interval has the wrong type. Should be a numpy array')
            self.t_start = t_interval[0]
            self.t_end = t_interval[-1]
            self.nt = len(t_interval)
            self.t = t_interval

    @property
    def vector_template(self) -> Vector:
        """
        Getter vector_template
        """
        return self._vector_template

    @vector_template.setter
    def vector_template(self, value: Vector) -> None:
        """
        Setter vector_template
        """
        self._vector_template = value

    @property
    def vector_t_start(self) -> Vector:
        """
        Getter vector_t_start
        """
        return self._vector_t_start

    @vector_t_start.setter
    def vector_t_start(self, value: Vector) -> None:
        """
        Getter vector_t_start
        """
        self._vector_t_start = value

    @abstractmethod
    def step(self, u_start: object, t_start: float, t_stop: float) -> object:
        """
        Time integration routine for the application
        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
