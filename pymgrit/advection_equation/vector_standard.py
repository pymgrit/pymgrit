import numpy as np
from scipy import linalg as la

from pymgrit.core import vector


class VectorStandard(vector.Vector):
    """
    """

    def __init__(self, size):
        super(VectorStandard, self).__init__()
        self.size = size
        self.vec = np.zeros(size)

    def __add__(self, other):
        tmp = VectorStandard(self.size)
        tmp.vec = self.vec + other.vec
        return tmp

    def __sub__(self, other):
        tmp = VectorStandard(self.size)
        tmp.vec = self.vec - other.vec
        return tmp

    def norm(self):
        return la.norm(self.vec)

    def init_zero(self):
        return VectorStandard(self.size)

    def init_rand(self):
        tmp = VectorStandard(self.size)
        tmp.vec = np.random.rand(self.size)
        return tmp
