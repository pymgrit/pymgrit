"""
Abstract vector class for the solution of one point in time.
Every vector class must inherit from this class.
Contains the spatial solution of one point in time
Required functions:
  - __add__
  - __sub__
  - norm
  - clone_zero
  - clone_rand
  - set_values
  - get_values
"""

from abc import ABC, abstractmethod


class Vector(ABC):
    """
    Abstract vector class for the solution of one point in time.
    Every vector class must inherit from this class.
    Contains the spatial solution of one point in time
    Required functions:
      - __add__
      - __sub__
      - norm
      - clone_zero
      - clone_rand
      - set_values
      - get_values
    """

    def __init__(self):
        pass

    @abstractmethod
    def __add__(self, other: '__class__') -> '__class__':
        """
        Addition
        :param other:
        """

    @abstractmethod
    def __sub__(self, other: '__class__') -> '__class__':
        """
        Subtraction
        :param other:
        """

    @abstractmethod
    def norm(self):
        """
        Norm of the solution vector
        """

    @abstractmethod
    def clone_rand(self):
        """
        Clones the solution vector with random values.
        """

    @abstractmethod
    def clone_zero(self):
        """
        Clones the solution vector with all zeros.
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
