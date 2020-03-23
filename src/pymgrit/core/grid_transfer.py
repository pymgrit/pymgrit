"""
Abstract grid transfer class for user-defined grid transfer classes.
Every user-defined grid transfer class must inherit from this class.
Allows for additional spatial coarsening and refinement between time levels.

Required functions:
  - restriction
  - interpolation
"""
from abc import ABC, abstractmethod

from pymgrit.core.vector import Vector


class GridTransfer(ABC):
    """
    Abstract grid transfer class for user-defined grid transfer classes.
    Every user-defined grid transfer class must inherit from this class.
    Allows for additional spatial coarsening and refinement between time levels.

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
        Spatial restriction.
        Receives the spatial solution at a point in time
        of one level and restricts it to the same point in time
        on the next coarser level.

        :param u: Approximate solution at a time point of one level
        :return: Restricted input vector u on next coarser time grid
        :rtype: Vector
        """

    @abstractmethod
    def interpolation(self, u: Vector) -> Vector:
        """
        Spatial interpolation.
        Receives the spatial solution at a point in time
        of one level and interpolates it to the same point in
        time on the next finer level.

        :param u: Approximate solution at a time point of one level
        :return: Interpolated input vector u on next finer time grid
        :rtype: Vector
        """
