"""
Tests grid transfer copy
"""
import numpy as np

from pymgrit.core.grid_transfer_copy import GridTransferCopy
from pymgrit.core.vector import Vector


class VectorSimple(Vector):
    def __init__(self):
        super(VectorSimple, self).__init__()

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def norm(self):
        pass

    def clone_zero(self):
        pass

    def clone_rand(self):
        pass

    def set_values(self, values):
        pass

    def get_values(self):
        pass

    def pack(self):
        pass

    def unpack(self):
        pass


def test_grid_transfer_copy_constructor():
    """
    Test constructor
    """
    grid_transfer_copy = GridTransferCopy()

    np.testing.assert_equal(True, isinstance(grid_transfer_copy, GridTransferCopy))


def test_grid_transfer_copy_restriction():
    """
    Test constructor
    """
    grid_transfer_copy = GridTransferCopy()

    np.testing.assert_equal(True, isinstance(grid_transfer_copy.restriction(VectorSimple()), VectorSimple))


def test_grid_transfer_copy_interpolation():
    """
    Test constructor
    """
    grid_transfer_copy = GridTransferCopy()

    np.testing.assert_equal(True, isinstance(grid_transfer_copy.interpolation(VectorSimple()), VectorSimple))
