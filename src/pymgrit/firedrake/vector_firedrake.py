"""
Vector class for Firedrake applications
"""

try:
    from firedrake import norm, Function
except ImportError as e:
    import sys

    sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

import numpy as np

from pymgrit.core.vector import Vector


class VectorFiredrake(Vector):
    """
    Vector class for Firedrake Function object
    """

    def __init__(self, values: Function):
        """
        Constructor.
        """

        super().__init__()
        if isinstance(values, Function):
            self.values = values.copy(deepcopy=True)
        else:
            raise Exception('Wrong datatype')

    def set_values(self, values):
        """
        Set vector data

        :param values: values for vector object
        """
        self.values = values

    def get_values(self):
        """
        Get vector data

        :return: values of vector object
        """
        return self.values

    def clone(self):
        """
        Initialize vector object with copied values

        :rtype: vector object with zero values
        """
        return VectorFiredrake(self.values)

    def clone_zero(self):
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """
        tmp = VectorFiredrake(self.values)
        tmp = tmp * 0
        return tmp

    def clone_rand(self):
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        tmp = VectorFiredrake(self.values)
        tmp_values = tmp.get_values()
        tmp_values.dat.data[:] = np.random.rand(*tmp_values.dat.data[:].shape)
        tmp.set_values(tmp_values)
        return tmp

    def __add__(self, other):
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        tmp = VectorFiredrake(self.values)
        tmp_value = tmp.get_values()
        tmp_value += other.get_values()
        tmp.set_values(tmp_value)
        return tmp

    def __sub__(self, other):
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        tmp = VectorFiredrake(self.values)
        tmp_value = tmp.get_values()
        tmp_value -= other.get_values()
        tmp.set_values(tmp_value)
        return tmp

    def __mul__(self, other):
        """
        Multiplication of a vector object and a float (self and other)

        :param other: object to be multiplied with self
        :return: difference of vector object self and input object other
        """
        tmp = VectorFiredrake(self.values)
        tmp_value = tmp.get_values()
        tmp_value *= other
        tmp.set_values(tmp_value)
        return tmp

    def norm(self):
        """
        Norm of a vector object

        :return: 2-norm of vector object
        """
        return norm(self.values)

    def unpack(self, values):
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.values.dat.data[:] = values

    def pack(self):
        """
        Pack data

        :return: values of vector object
        """
        return self.values.dat.data[:]
