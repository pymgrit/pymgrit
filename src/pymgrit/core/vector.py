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
    def __mul__(self, other: '__class__') -> '__class__':
        """
        Multiplication of one vector object with a float (self and other)

        :param other: object to be multiplied with self
        :return: multiplication of vector object self and input object other
        """

    @abstractmethod
    def norm(self):
        """
        Norm of a vector object
        """

    @abstractmethod
    def clone(self):
        """
        Initialize vector object with same values
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

    def __rmul__(self, other):
        """
        Right multiplication of one vector object with a float (self and other). Based on function __mul__.

        :param other: object to be multiplied with self
        :return: multiplication of vector object self and input object other
        """

        return self * other

    def __imul__(self, other):
        """
        In-place multiplication based on function __mul__

        :param other: object to be multiplied with self
        :return: multiplication of vector object self and input vector object other
        """

        return self * other

    def __iadd__(self, other):
        """
        In-place addition based on function __add__

        :param other: object to be multiplied with self
        :return: sum of vector object self and input vector object other
        """

        return self + other

    def __isub__(self, other):
        """
        In-place subtraction based on function __sub__

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input vector object other
        """

        return self - other
