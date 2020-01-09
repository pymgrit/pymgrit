import numpy as np
from scipy import linalg as la

from pymgrit.core import vector


class VectorStandardBDF2(vector.Vector):
    """
    Solution vector for two points
    """

    def __init__(self, size):
        """
        Constructor. One solution vector contains two time points.
        :param size:
        """
        super(VectorStandardBDF2, self).__init__()
        self.size = size
        self.vec_first_time_point = np.zeros(size)
        self.vec_second_time_point = np.zeros(size)

    def __add__(self, other):
        """
        Addition
        :param other:
        :return:
        """
        tmp = VectorStandardBDF2(self.size)
        tmp.vec_first_time_point = self.vec_first_time_point + other.vec_first_time_point
        tmp.vec_second_time_point = self.vec_second_time_point + other.vec_second_time_point
        return tmp

    def __sub__(self, other):
        """
        Subtraction
        :param other:
        :return:
        """
        tmp = VectorStandardBDF2(self.size)
        tmp.vec_first_time_point = self.vec_first_time_point - other.vec_first_time_point
        tmp.vec_second_time_point = self.vec_second_time_point - other.vec_second_time_point
        return tmp

    def norm(self):
        """
        Norm
        :return:
        """
        return la.norm(np.append(self.vec_first_time_point, self.vec_second_time_point))

    def init_zero(self):
        """
        Initial solution vector with all zeros
        :rtype: object
        """
        return VectorStandardBDF2(self.size)

    def init_rand(self):
        """
        Initial solution vector with random values
        :rtype: object
        """
        tmp = VectorStandardBDF2(self.size)
        tmp.vec_first_time_point = np.random.rand(self.size)
        tmp.vec_second_time_point = np.random.rand(self.size)
        return tmp
