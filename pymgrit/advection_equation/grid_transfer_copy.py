import copy

from pymgrit.core import grid_transfer


class GridTransferCopy(grid_transfer.GridTransfer):
    """
    Standard grid transfer operator. Copies the values without changes
    """

    def __init__(self):
        super(GridTransferCopy, self).__init__()
        """
        """

    def restriction(self, u):
        return copy.deepcopy(u)

    def interpolation(self, u):
        return copy.deepcopy(u)
