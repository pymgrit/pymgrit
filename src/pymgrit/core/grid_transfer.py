"""
Abstract class for grid transfers. Each grid transfer has to be a child
"""
from abc import ABC, abstractmethod


class GridTransfer(ABC):
    """
    Abstract class for grid transfers. Each grid transfer has to be a child
    """

    def __init__(self):
        """
        Constructor
        """

    @abstractmethod
    def restriction(self, u):
        """
        Spatial restriction between two grids.
        Gets solutions of one grid and transforms it to another one
        :param u: Start solution
        """

    @abstractmethod
    def interpolation(self, u):
        """
        Spatial interpolation between two grids.
        Gets solutions of one grid and transforms it to another one
        :param u: Start solution
        """
