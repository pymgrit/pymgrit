import numpy as np
from abstract_classes import vector
from numpy import linalg as la


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

    def clone_zeros(self):
        return VectorStandard(self.size)
