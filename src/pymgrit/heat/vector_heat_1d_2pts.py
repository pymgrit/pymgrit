"""
Vector class for 1D heat problem

Note: values at two consecutive time points are grouped as pairs
"""

import numpy as np

from pymgrit.core.vector import Vector


class VectorHeat1D2Pts(Vector):
    """
    Vector class for grouping values at two consecutive time points
    """

    def __init__(self, size, dtau):
        """
        Constructor.
        One vector object contains values at two consecutive time points and spacing between these time points.

        :param size: number of spatial degrees of freedom
        :param dtau: time-step size within pair
        """
        super().__init__()
        self.size = size
        self.dtau = dtau
        self.values_first_time_point = np.zeros(size)
        self.values_second_time_point = np.zeros(size)

    def __add__(self, other):
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        tmp = VectorHeat1D2Pts(self.size, self.dtau)
        first_self, second_self, dtau_self = self.get_values()
        first_other, second_other, dtau_other = other.get_values()
        tmp.set_values(first_self + first_other, second_self + second_other, dtau_self)
        return tmp

    def __sub__(self, other):
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        tmp = VectorHeat1D2Pts(self.size, self.dtau)
        first_self, second_self, dtau_self = self.get_values()
        first_other, second_other, dtau_other = other.get_values()
        tmp.set_values(first_self - first_other, second_self - second_other, dtau_self)
        return tmp

    def __mul__(self, other):
        """
        Multiplication of one vector object with a float (self and other)

        :param other: object to be multiplied with self
        :return: difference of vector object self and input object other
        """
        tmp = VectorHeat1D2Pts(self.size, self.dtau)
        first_self, second_self, dtau_self = self.get_values()
        tmp.set_values(first_self * other, second_self * other, dtau_self)
        return tmp

    def norm(self):
        """
        Norm of a vector object

        :return: 2-norm of vector object
        """
        return np.linalg.norm(np.append(self.values_first_time_point, self.values_second_time_point))

    def clone(self):
        """
        Clone vector object

        :return: vector object with zero values
        """
        tmp = VectorHeat1D2Pts(self.size, self.dtau)
        tmp.set_values(self.values_first_time_point, self.values_second_time_point, self.dtau)
        return tmp

    def clone_zero(self):
        """
        Initialize vector object with zeros

        :return: vector object with zero values
        """
        return VectorHeat1D2Pts(self.size, self.dtau)

    def clone_rand(self):
        """
        Initialize vector object with random values

        :return: vector object with random values
        """
        tmp = VectorHeat1D2Pts(self.size, self.dtau)
        tmp.set_values(np.random.rand(self.size), np.random.rand(self.size), self.dtau)
        return tmp

    def get_values(self):
        """
        Get vector data

        :return: tuple of values of member variables
        """
        return self.values_first_time_point, self.values_second_time_point, self.dtau

    def set_values(self, first_time_point, second_time_point, dtau):
        """
        Set vector data

        :param first_time_point: values for first time point
        :param second_time_point: values for second time point
        :param dtau: time-step size within pair
        """
        self.values_first_time_point = first_time_point
        self.values_second_time_point = second_time_point
        self.dtau = dtau

    def pack(self):
        """
        Pack data

        :return: values of vector object
        """
        return np.array([self.values_first_time_point, self.values_second_time_point])

    def unpack(self, values):
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.values_first_time_point = values[0]
        self.values_second_time_point = values[1]
