"""
Abstract vector class for user-defined vector classes that
hold information of a single time point.
Every user-defined vector class must inherit from this class.

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
    Abstract vector class for user-defined vector classes that
    hold information of a single time point.
    Every user-defined vector class must inherit from this class.

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
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input vector object other
        """

    @abstractmethod
    def __sub__(self, other: '__class__') -> '__class__':
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input vector object other
        """

    @abstractmethod
    def norm(self):
        """
        Norm of a vector object
        """

    @abstractmethod
    def clone_rand(self):
        """
        Initialize vector object with random values
        """

    @abstractmethod
    def clone_zero(self):
        """
        Initialize vector object with zeros
        """

    @abstractmethod
    def set_values(self, *args, **kwargs):
        """
        Set vector data
        """

    @abstractmethod
    def get_values(self, *args, **kwargs):
        """
        Get vector data
        """

    @abstractmethod
    def pack(self, *args, **kwargs):
        """
        Specifying communication data
        """

    @abstractmethod
    def unpack(self, *args, **kwargs):
        """
        Unpacking communication data
        """
