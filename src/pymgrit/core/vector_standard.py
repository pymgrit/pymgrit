"""
Standard solution vector. Contains one numpy array
"""

import numpy as np

from pymgrit.core.vector import Vector


class VectorStandard(Vector):
    """
    Standard solution vector. Contains one numpy array
    """

    def __init__(self, size):
        """
        Constructor
        :param size:
        """
        super(VectorStandard, self).__init__()
        self.size = size
        self.vec = np.zeros(size)

    def __add__(self, other):
        """
        Addition
        :rtype: object
        """
        tmp = VectorStandard(self.size)
        tmp.vec = self.vec + other.vec
        return tmp

    def __sub__(self, other):
        """
        Subtraction
        :rtype: object
        """
        tmp = VectorStandard(self.size)
        tmp.vec = self.vec - other.vec
        return tmp

    def norm(self):
        """
        Norm
        :rtype: object
        """
        return np.linalg.norm(self.vec)

    def init_zero(self):
        """
        Initial solution vector with all zeros
        :rtype: object
        """
        return VectorStandard(self.size)

    def init_rand(self):
        """
        Initial solution vector with random values
        :rtype: object
        """
        tmp = VectorStandard(self.size)
        tmp.vec = np.random.rand(self.size)
        return tmp
