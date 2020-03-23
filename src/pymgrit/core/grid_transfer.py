"""
Abstract grid transfer class.
Every grid transfer class must inherit from this class.
Transports the spatial solutions between the levels.
Required functions:
  - restriction
  - interpolation
"""
from abc import ABC, abstractmethod

from pymgrit.core.vector import Vector


class GridTransfer(ABC):
    """
    Abstract grid transfer class.
    Every grid transfer class must inherit from this class.
    Transports the spatial solutions between the levels.
    Required functions:
      - restriction
      - interpolation
    """

    def __init__(self):
        """
        Constructor
        """

    @abstractmethod
    def restriction(self, u: Vector) -> Vector:
        """
        Receives the spatial solution from a point in time
        of one level and transforms it into the solution of
        the same point in time on the next lower level.
        :rtype: Vector
        :param u: Solution on higher level
        """

    @abstractmethod
    def interpolation(self, u: Vector) -> Vector:
        """
        Receives the spatial solution from a point in time
        of one level and transforms it into the solution of
        the same point in time on the next higher level.
        :rtype: Vector
        :param u: Solution on coarser level
        """
