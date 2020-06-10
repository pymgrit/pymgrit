"""
Vector for the induction
machine model "im_3kW". (https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines)
"""
from typing import List
import numpy as np

from pymgrit.core.vector import Vector


class VectorMachine(Vector):
    """
    Vector for the induction machine im_3kW
    """

    def __init__(self, u_front_size: int, u_middle_size: int, u_back_size: int) -> None:
        """
        Constructor
        :param u_front_size:
        :param u_middle_size:
        :param u_back_size:
        """
        super().__init__()
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
        self.tr = 0

    def __add__(self, other: 'VectorMachine') -> 'VectorMachine':
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
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
        tmp.tr = self.tr + other.tr
        return tmp

    def __sub__(self, other: 'VectorMachine') -> 'VectorMachine':
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
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
        tmp.tr = self.tr - other.tr
        return tmp

    def norm(self) -> float:
        """
        Norm of a vector object

        :return: 2-norm of vector object
        """
        tmp = np.append(np.append(self.u_front, self.u_middle), self.u_back)
        # tmp = np.append(tmp, [self.jl, self.ua, self.ub, self.uc, self.ia, self.ib, self.ic, self.tr])
        return np.linalg.norm(tmp)

    def clone(self):
        """
        Initial solution vector with all zeros
        :rtype: object
        """
        tmp = VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))
        tmp.unpack(self.pack())
        return tmp

    def clone_zero(self) -> 'VectorMachine':
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """
        return VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))

    def clone_rand(self) -> 'VectorMachine':
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        return VectorMachine(len(self.u_front), len(self.u_middle), len(self.u_back))

    def get_values(self) -> 'np.ndarray':
        """
        Get vector data

        :return: values of vector object
        """
        return np.append(np.append(self.u_front, self.u_middle), self.u_back)

    def set_values(self, values: np.ndarray, jl: float, ia: float, ib: float, ic: float, ua: float, ub: float,
                   uc: float, tr: float) -> None:
        """
        Set vector data
        """
        self.u_front = values[:self.u_front_size]
        self.u_middle = values[self.u_front_size:-self.u_back_size]
        self.u_back = values[-self.u_back_size:]
        self.jl = jl
        self.ia = ia
        self.ib = ib
        self.ic = ic
        self.ua = ua
        self.ub = ub
        self.uc = uc
        self.tr = tr

    def pack(self) -> List:
        """
        Pack data

        :return: values of vector object
        """
        send_obj = [self.u_front, self.u_middle, self.u_back, self.jl, self.ia, self.ib, self.ic, self.ua, self.ub,
                    self.uc, self.tr]
        return send_obj

    def unpack(self, values: List) -> None:
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.u_front = values[0]
        self.u_middle = values[1]
        self.u_back = values[2]
        self.jl = values[3]
        self.ia = values[4]
        self.ib = values[5]
        self.ic = values[6]
        self.ua = values[7]
        self.ub = values[8]
        self.uc = values[9]
        self.tr = values[10]
