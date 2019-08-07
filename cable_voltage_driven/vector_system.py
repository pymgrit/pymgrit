import numpy as np
from abstract_classes import vector
from scipy import linalg as la


class VectorSystem(vector.Vector):
    """
    """

    def __init__(self, size):
        super(VectorSystem, self).__init__()
        self.size = size
        self.mvp = np.zeros(size)
        self.current = 0

    def __add__(self, other):
        tmp = VectorSystem(self.size)
        tmp.mvp = self.mvp + other.mvp
        tmp.current = self.current + other.current
        return tmp

    def __sub__(self, other):
        tmp = VectorSystem(self.size)
        tmp.mvp = self.mvp - other.mvp
        tmp.current = self.current - other.current
        return tmp

    def norm(self):
        tmp = np.zeros(self.size + 1)
        tmp[:-1] = self.mvp
        tmp[-1] = self.current
        return la.norm(tmp)

    def clone_zeros(self):
        return VectorSystem(self.size)