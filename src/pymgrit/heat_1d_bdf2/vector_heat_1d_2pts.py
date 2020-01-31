import numpy as np

from pymgrit.core.vector import Vector


class VectorHeat1D2Pts(Vector):
    """
    Solution vector for two points
    """

    def __init__(self, size):
        """
        Constructor. One solution vector contains two time points.
        :param size:
        """
        super(VectorHeat1D2Pts, self).__init__()
        self.size = size
        self.values_first_time_point = np.zeros(size)
        self.values_second_time_point = np.zeros(size)

    def __add__(self, other):
        """
        Addition
        :param other:
        :return:
        """
        tmp = VectorHeat1D2Pts(self.size)
        first_self, second_self = self.get_values()
        first_other, second_other = other.get_values()
        tmp.set_values(first_self + first_other, second_self + second_other)
        return tmp

    def __sub__(self, other):
        """
        Subtraction
        :param other:
        :return:
        """
        tmp = VectorHeat1D2Pts(self.size)
        first_self, second_self = self.get_values()
        first_other, second_other = other.get_values()
        tmp.set_values(first_self - first_other, second_self - second_other)
        return tmp

    def norm(self):
        """
        Norm
        :return:
        """
        return np.linalg.norm(np.append(self.values_first_time_point, self.values_second_time_point))

    def clone_zero(self):
        """
        Initial solution vector with all zeros
        :rtype: object
        """
        return VectorHeat1D2Pts(self.size)

    def clone_rand(self):
        """
        Initial solution vector with random values
        :rtype: object
        """
        tmp = VectorHeat1D2Pts(self.size)
        tmp.set_values(np.random.rand(self.size), np.random.rand(self.size))
        return tmp

    def get_values(self):
        return self.values_first_time_point, self.values_second_time_point

    def set_values(self, first_time_point, second_time_point):
        self.values_first_time_point = first_time_point
        self.values_second_time_point = second_time_point
