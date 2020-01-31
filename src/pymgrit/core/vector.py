"""
Abstract class for one solution point. Each solution data structure has to be a child
"""

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

    @abstractmethod
    def __sub__(self, other):
        """
        Standard subtraction
        :param other:
        """

    @abstractmethod
    def norm(self):
        """
        Norm of the solution construct
        """

    @abstractmethod
    def clone_rand(self):
        """
        Clones the solution construct with random values.
        """

    @abstractmethod
    def clone_zero(self):
        """
        Clones the solution construct with all zeros.
        """

    @abstractmethod
    def set_values(self, *args, **kwargs):
        """
        Sets the values
        """

    @abstractmethod
    def get_values(self, *args, **kwargs):
        """
        Gets the values
        """
