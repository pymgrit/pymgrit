"""
Standard grid transfer operator.
Copies the spatial solutions from one level to another.
"""

import copy

from pymgrit.core import grid_transfer
from pymgrit.core.vector import Vector


class GridTransferCopy(grid_transfer.GridTransfer):
    """
    Standard grid transfer.
    Copies the spatial solutions from one level to another.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(GridTransferCopy, self).__init__()

    def restriction(self, u: Vector) -> Vector:
        """
        Copies the spatial solutions from one point in time
        :param u: Solution of one point in time
        :return:  Solution of the same point in time on the next lower level
        :rtype: Vector
        """
        return copy.deepcopy(u)

    def interpolation(self, u: Vector) -> Vector:
        """
        Copies the spatial solutions from one point in time
        :param u: Solution of one point in time
        :return:  Solution of the same point in time on the next higher level
        :rtype: Vector
        """
        return copy.deepcopy(u)
