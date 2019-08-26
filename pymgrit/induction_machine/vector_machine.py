import numpy as np
from pymgrit.core import vector
from scipy import linalg as la


class VectorMachine(vector.Vector):
    """
    """

    def __init__(self, u_front_size, u_middle_size, u_back_size):
        super(VectorMachine, self).__init__()
        self.u_front_size = u_front_size
        self.u_middle_size = u_middle_size
        self.u_back_size = u_back_size
        self.u_front = np.zeros(u_front_size)
        self.u_back = np.zeros(u_back_size)
        self.u_middle = np.zeros(u_middle_size)
        self.jl = 0
        self.ua = 0
        self.ub = 0
        self.uc = 0
        self.ia = 0
        self.ib = 0
        self.ic = 0

    def __add__(self, other):
        tmp = VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))
        tmp.u_front = self.u_front + other.u_front
        tmp.u_back = self.u_back + other.u_back
        tmp.u_middle = self.u_middle + other.u_middle
        tmp.jl = self.jl + other.jl
        tmp.ia = self.ia + other.ia
        tmp.ib = self.ib + other.ib
        tmp.ic = self.ic + other.ic
        tmp.ua = self.ua + other.ua
        tmp.ub = self.ub + other.ub
        tmp.uc = self.uc + other.uc
        return tmp

    def __sub__(self, other):
        tmp = VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))
        tmp.u_front = self.u_front - other.u_front
        tmp.u_back = self.u_back - other.u_back
        tmp.u_middle = self.u_middle - other.u_middle
        tmp.jl = self.jl - other.jl
        tmp.ia = self.ia - other.ia
        tmp.ib = self.ib - other.ib
        tmp.ic = self.ic - other.ic
        tmp.ua = self.ua - other.ua
        tmp.ub = self.ub - other.ub
        tmp.uc = self.uc - other.uc
        return tmp

    def norm(self):
        tmp = np.append(self.u_front, self.u_middle)
        tmp = np.append(tmp, self.u_back)
        tmp = np.append(tmp, [self.jl, self.ua, self.ub, self.uc, self.ia, self.ib, self.ic])
        return la.norm(tmp)

    def init_zero(self):
        return VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))

    def init_rand(self):
        tmp = VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))
        self.u_front = np.random.rand(len(self.u_front))
        self.u_back = np.random.rand(len(self.u_back))
        self.u_middle = np.random.rand(len(self.u_middle))
        self.jl = np.random.rand(1)[0]
        self.ua = np.random.rand(1)[0]
        self.ub = np.random.rand(1)[0]
        self.uc = np.random.rand(1)[0]
        self.ia = np.random.rand(1)[0]
        self.ib = np.random.rand(1)[0]
        self.ic = np.random.rand(1)[0]
        return tmp
