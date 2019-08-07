from abc import ABC, abstractmethod


class Vector(ABC):
    """
    Abstract class for one solution point. Each solution data structure has to be a child
    """

    def __init__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        """
        Standard addition
        :param other:
        """
        pass

    @abstractmethod
    def __sub__(self, other):
        """
        Standard subtraction
        :param other:
        """
        pass

    @abstractmethod
    def norm(self):
        """
        Norm of the solution construct
        """
        pass

    @abstractmethod
    def clone_zeros(self):
        """
        Clones the solution construct with all zeros.
        """
        pass
