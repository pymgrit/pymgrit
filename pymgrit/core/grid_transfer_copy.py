"""
Standard grid transfer. Copies values
"""

import copy

from pymgrit.core import grid_transfer


class GridTransferCopy(grid_transfer.GridTransfer):
    """
    Standard grid transfer. Copies values
    """

    def __init__(self):
        """
        Constructor.
        :rtype: object
        """
        super(GridTransferCopy, self).__init__()

    def restriction(self, u):
        """
        Restrict u
        :rtype: object
        """
        return copy.deepcopy(u)

    def interpolation(self, u):
        """
        Interpolate u
        :rtype: object
        """
        return copy.deepcopy(u)
