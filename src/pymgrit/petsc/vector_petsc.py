"""
Vector class for PETSc applications
"""

import copy
import numpy as np

from pymgrit.core.vector import Vector as PymgritVector

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


class VectorPetsc(PymgritVector):
    """
    Vector class for PETSc vectors
    """

    def __init__(self, values: PETSc.Vec) -> None:
        """
        Constructor.

        :param values: PETSc.Vec with approximation
        """
        if isinstance(values, PETSc.Vec):
            self.values = copy.deepcopy(values)
        else:
            raise Exception('Wrong datatype')

    def __add__(self, other: '__class__') -> '__class__':
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        return VectorPetsc(self.get_values() + other.get_values())

    def __sub__(self, other: '__class__') -> '__class__':
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        return VectorPetsc(self.get_values() - other.get_values())

    def __mul__(self, other) -> '__class__':
        """
        Multiplication of a vector object and a float (self and other)

        :param other: object to be multiplied with self
        :return: difference of vector object self and input object other
        """
        return VectorPetsc(self.get_values() * other)

    def norm(self) -> float:
        """
        Norm of a vector object

        :return: Frobenius-norm of vector object
        """
        return self.values.norm(PETSc.NormType.FROBENIUS)

    def clone(self) -> '__class__':
        """
        Initialize vector object with copied values

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values())

    def clone_zero(self) -> '__class__':
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values() * 0)

    def clone_rand(self) -> '__class__':
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        # TODO
        return VectorPetsc(self.get_values())

    def set_values(self, values: PETSc.Vec) -> None:
        """
        Set vector data

        :param values: values for vector object
        """
        self.values = values

    def get_values(self) -> PETSc.Vec:
        """
        Get vector data

        :return: values of vector object
        """
        return self.values

    def pack(self) -> np.ndarray:
        """
        Pack data

        :return: values of vector object
        """
        return self.values.getArray()

    def unpack(self, values: np.ndarray) -> None:
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.values.setArray(values)
