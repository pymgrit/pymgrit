"""
Tests advection_1d
"""
import numpy as np

from pymgrit.advection.advection_1d import Advection1D
from pymgrit.advection.advection_1d import VectorAdvection1D


def test_advection_1d_constructor():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    nx = 11
    c = 1
    advection_1d = Advection1D(c=c, x_start=x_start, x_end=x_end, nx=nx, t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(advection_1d.x_start, x_start)
    np.testing.assert_equal(advection_1d.x_end, x_end)
    np.testing.assert_equal(advection_1d.nx, nx - 1)
    np.testing.assert_almost_equal(advection_1d.dx, 0.1)
    np.testing.assert_equal(advection_1d.x, np.linspace(x_start, x_end, nx)[0:-1])

    np.testing.assert_equal(True, isinstance(advection_1d.vector_template, VectorAdvection1D))
    np.testing.assert_equal(True, isinstance(advection_1d.vector_t_start, VectorAdvection1D))
    np.testing.assert_equal(advection_1d.vector_t_start.get_values(),
                            np.exp(-np.linspace(x_start, x_end, nx)[0:-1] ** 2))


def test_advection_1d_step():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    nx = 6
    c = 1
    advection_1d = Advection1D(c=c, x_start=x_start, x_end=x_end, nx=nx, t_start=0, t_stop=1, nt=11)
    advection_1d_res = advection_1d.step(u_start=advection_1d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(advection_1d_res.get_values(),
                                   np.array([0.868043, 0.92987396, 0.87805385, 0.75780217, 0.604129]))


def test_vector_advection_1d_constructor():
    """
    Test constructor
    """
    vector_advection_1d = VectorAdvection1D(3)
    np.testing.assert_equal(vector_advection_1d.values[0], 0)
    np.testing.assert_equal(vector_advection_1d.values[1], 0)
    np.testing.assert_equal(vector_advection_1d.values[2], 0)


def test_vector_advection_1d_add():
    """
    Test __add__
    """
    vector_advection_1d_1 = VectorAdvection1D(5)
    vector_advection_1d_1.values = np.ones(5)
    vector_advection_1d_2 = VectorAdvection1D(5)
    vector_advection_1d_2.values = 2 * np.ones(5)

    vector_advection_1d_res = vector_advection_1d_1 + vector_advection_1d_2
    np.testing.assert_equal(vector_advection_1d_res.values, 3 * np.ones(5))

    vector_advection_1d_1 += vector_advection_1d_2
    np.testing.assert_equal(vector_advection_1d_1.values, 3 * np.ones(5))


def test_vector_advection_1d_sub():
    """
    Test __sub__
    """
    vector_advection_1d_1 = VectorAdvection1D(5)
    vector_advection_1d_1.values = np.ones(5)
    vector_advection_1d_2 = VectorAdvection1D(5)
    vector_advection_1d_2.values = 2 * np.ones(5)

    vector_advection_1d_res = vector_advection_1d_2 - vector_advection_1d_1
    np.testing.assert_equal(vector_advection_1d_res.values, np.ones(5))

    vector_advection_1d_2 -= vector_advection_1d_1
    np.testing.assert_equal(vector_advection_1d_2.values, np.ones(5))

def test_vector_advection_1d_mul():
    """
    Test __mul__
    """
    vector_advection_1d_1 = VectorAdvection1D(5)
    vector_advection_1d_1.values = np.ones(5)

    vector_advection_1d_res = vector_advection_1d_1 * 5
    np.testing.assert_equal(vector_advection_1d_res.values, np.ones(5)*5)

    vector_advection_1d_res = 7 * vector_advection_1d_1
    np.testing.assert_equal(vector_advection_1d_res.values, np.ones(5)*7)

    vector_advection_1d_1 *= 9
    np.testing.assert_equal(vector_advection_1d_1.values, np.ones(5)*9)

def test_vector_advection_1d_norm():
    """
    Test norm()
    """
    vector_advection_1d = VectorAdvection1D(5)
    vector_advection_1d.values = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(np.linalg.norm(np.array([1, 2, 3, 4, 5])), vector_advection_1d.norm())


def test_vector_advection_1d_clone_zero():
    """
    Test clone_zero()
    """
    vector_advection_1d = VectorAdvection1D(2)

    vector_advection_1d_clone = vector_advection_1d.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_advection_1d_clone, VectorAdvection1D))

    np.testing.assert_equal(vector_advection_1d_clone.values, np.zeros(2))
    np.testing.assert_equal(len(vector_advection_1d_clone.values), 2)


def test_vector_advection_1d_clone_rand():
    """
    Test clone_rand()
    """
    vector_advection_1d = VectorAdvection1D(2)

    vector_advection_1d_clone = vector_advection_1d.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_advection_1d_clone, VectorAdvection1D))
    np.testing.assert_equal(len(vector_advection_1d_clone.values), 2)


def test_vector_advection_1d_set_values():
    """
    Test the set_values()
    """
    vector_advection_1d = VectorAdvection1D(2)
    vector_advection_1d.set_values(np.array([1, 2]))
    np.testing.assert_equal(vector_advection_1d.values, np.array([1, 2]))


def test_vector_advection_1d_get_values():
    """
    Test get_values()
    """
    vector_advection_1d = VectorAdvection1D(2)
    np.testing.assert_equal(vector_advection_1d.get_values(), np.zeros(2))
