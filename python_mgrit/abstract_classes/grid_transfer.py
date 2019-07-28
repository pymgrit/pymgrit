from abc import ABC, abstractmethod


class GridTransfer(ABC):
    """
    Abstract class for grid transfers. Each grid transfer has to be a child
    """

    def __init__(self):
        pass

    @abstractmethod
    def restriction(self, u):
        pass

    @abstractmethod
    def interpolation(self, u):
        pass
