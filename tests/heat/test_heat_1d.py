"""
Tests heat_1d
"""
import numpy as np

from pymgrit.heat.heat_1d import Heat1D
from pymgrit.heat.heat_1d import VectorHeat1D


def test_heat_1d_constructor():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    nx = 11
    a = 1
    heat_1d = Heat1D(a=a, x_start=x_start, x_end=x_end, nx=nx, t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(heat_1d.x_start, x_start)
    np.testing.assert_equal(heat_1d.x_end, x_end)
    np.testing.assert_equal(heat_1d.nx, nx - 2)
    np.testing.assert_almost_equal(heat_1d.dx, 0.1)
    np.testing.assert_equal(heat_1d.x, np.linspace(x_start, x_end, nx)[1:-1])

    np.testing.assert_equal(True, isinstance(heat_1d.vector_template, VectorHeat1D))
    np.testing.assert_equal(True, isinstance(heat_1d.vector_t_start, VectorHeat1D))
    np.testing.assert_equal(heat_1d.vector_t_start.get_values(), np.zeros(9))


def test_heat_1d_step():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    nx = 6
    a = 1
    heat_1d = Heat1D(a=a, init_cond=lambda x: 2 * x, x_start=x_start, x_end=x_end, nx=nx, t_start=0, t_stop=1, nt=11)
    heat_1d_res = heat_1d.step(u_start=heat_1d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(heat_1d_res.get_values(), np.array([0.28164, 0.51593599, 0.63660638, 0.53191933]))


def test_vector_heat_1d_constructor():
    """
    Test constructor
    """
    vector_heat_1d = VectorHeat1D(3)
    np.testing.assert_equal(vector_heat_1d.values[0], 0)
    np.testing.assert_equal(vector_heat_1d.values[1], 0)
    np.testing.assert_equal(vector_heat_1d.values[2], 0)


def test_vector_heat_1d_add():
    """
    Test __add__
    """
    vector_heat_1d_1 = VectorHeat1D(5)
    vector_heat_1d_1.values = np.ones(5)
    vector_heat_1d_2 = VectorHeat1D(5)
    vector_heat_1d_2.values = 2 * np.ones(5)

    vector_heat_1d_res = vector_heat_1d_1 + vector_heat_1d_2
    np.testing.assert_equal(vector_heat_1d_res.values, 3 * np.ones(5))


def test_vector_heat_1d_sub():
    """
    Test __sub__
    """
    vector_heat_1d_1 = VectorHeat1D(5)
    vector_heat_1d_1.values = np.ones(5)
    vector_heat_1d_2 = VectorHeat1D(5)
    vector_heat_1d_2.values = 2 * np.ones(5)

    vector_heat_1d_res = vector_heat_1d_2 - vector_heat_1d_1
    np.testing.assert_equal(vector_heat_1d_res.values, np.ones(5))

def test_vector_heat_1d_mul():
    """
    Test __mul__
    """
    vector_heat_1d_1 = VectorHeat1D(5)
    vector_heat_1d_1.values = np.ones(5)

    vector_heat_1d_res = vector_heat_1d_1 * 3
    np.testing.assert_equal(vector_heat_1d_res.values, np.ones(5)*3)

def test_vector_heat_1d_norm():
    """
    Test norm()
    """
    vector_heat_1d = VectorHeat1D(5)
    vector_heat_1d.values = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(np.linalg.norm(np.array([1, 2, 3, 4, 5])), vector_heat_1d.norm())


def test_vector_heat_1d_clone_zero():
    """
    Test clone_zero()
    """
    vector_heat_1d = VectorHeat1D(2)

    vector_heat_1d_clone = vector_heat_1d.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_clone, VectorHeat1D))

    np.testing.assert_equal(vector_heat_1d_clone.values, np.zeros(2))
    np.testing.assert_equal(len(vector_heat_1d_clone.values), 2)


def test_vector_heat_1d_clone_rand():
    """
    Test clone_rand()
    """
    vector_heat_1d = VectorHeat1D(2)

    vector_heat_1d_clone = vector_heat_1d.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_clone, VectorHeat1D))
    np.testing.assert_equal(len(vector_heat_1d_clone.values), 2)


def test_vector_heat_1d_set_values():
    """
    Test the set_values()
    """
    vector_heat_1d = VectorHeat1D(2)
    vector_heat_1d.set_values(np.array([1, 2]))
    np.testing.assert_equal(vector_heat_1d.values, np.array([1, 2]))


def test_vector_heat_1d_get_values():
    """
    Test get_values()
    """
    vector_heat_1d = VectorHeat1D(2)
    np.testing.assert_equal(vector_heat_1d.get_values(), np.zeros(2))
