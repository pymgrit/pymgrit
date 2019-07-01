from abc import ABC, abstractmethod


class GridTransfer(ABC):
    """
    """

    def __init__(self):
        """
        """

    @abstractmethod
    def restriction(self, u):
        pass

    @abstractmethod
    def interpolation(self, u):
        pass
