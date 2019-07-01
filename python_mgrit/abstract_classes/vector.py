from abc import ABC, abstractmethod


class Vector(ABC):
    """
    """

    def __init__(self):
        """
        """

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def norm(self):
        pass

    @abstractmethod
    def clone_zeros(self):
        pass
