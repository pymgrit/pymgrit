"""
Standard grid transfer operator.
Copies the spatial solution from one level to another.
"""

import copy

from pymgrit.core import grid_transfer
from pymgrit.core.vector import Vector


class GridTransferCopy(grid_transfer.GridTransfer):
    """
    Standard grid transfer.
    Copies the spatial solution from one level to another.
    This function is called for every time point.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(GridTransferCopy, self).__init__()

    def restriction(self, u: Vector) -> Vector:
        """
        Copies the spatial solution at one point in time
        of one level and restricts it to the same point in time
        on the next coarser level.

        :param u: Approximate solution at a time point of one level
        :return: Input vector u at the same time point on next coarser time grid
        :rtype: Vector
        """
        return u.clone()

    def interpolation(self, u: Vector) -> Vector:
        """
        Copies the spatial solution at one point in time
        of one level and restricts it to the same point in time
        on the next finer level.

        :param u: Approximate solution at a time point of one level
        :return: Input vector u at the same time point on next finer time grid
        :rtype: Vector
        """
        return u.clone()
