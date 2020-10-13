"""
Tests brusselator
"""
import numpy as np

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.brusselator.brusselator import VectorBrusselator


def test_brusselator_constructor():
    """
    Test constructor
    """
    brusselator = Brusselator(t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(brusselator.a, 1)
    np.testing.assert_equal(brusselator.b, 3)

    np.testing.assert_equal(True, isinstance(brusselator.vector_template, VectorBrusselator))
    np.testing.assert_equal(True, isinstance(brusselator.vector_t_start, VectorBrusselator))
    np.testing.assert_equal(brusselator.vector_t_start.get_values(), np.array([0, 1]))


def test_brusselator_step():
    """
    Test step()
    """
    brusselator = Brusselator(t_start=0, t_stop=1, nt=11)
    brusselator_res = brusselator.step(u_start=VectorBrusselator(), t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(brusselator_res.get_values(), np.array([0.08240173, 0.01319825]))


def test_vector_brusselator_constructor():
    """
    Test constructor
    """
    vector_brusselator = VectorBrusselator()
    np.testing.assert_equal(vector_brusselator.value[0], 0)
    np.testing.assert_equal(vector_brusselator.value[1], 0)


def test_vector_brusselator_add():
    """
    Test __add__
    """
    vector_brusselator_1 = VectorBrusselator()
    vector_brusselator_1.value = np.ones(2)
    vector_brusselator_2 = VectorBrusselator()
    vector_brusselator_2.value = 2 * np.ones(2)

    vector_brusselator_res = vector_brusselator_1 + vector_brusselator_2
    np.testing.assert_equal(vector_brusselator_res.value, 3 * np.ones(2))


def test_vector_brusselator_sub():
    """
    Test __sub__
    """
    vector_brusselator_1 = VectorBrusselator()
    vector_brusselator_1.value = np.ones(2)
    vector_brusselator_2 = VectorBrusselator()
    vector_brusselator_2.value = 2 * np.ones(2)

    vector_brusselator_res = vector_brusselator_2 - vector_brusselator_1
    np.testing.assert_equal(vector_brusselator_res.value, np.ones(2))

def test_vector_brusselator_mul():
    """
    Test __mul__
    """
    vector_brusselator_1 = VectorBrusselator()
    vector_brusselator_1.value = np.ones(2)

    vector_brusselator_res = vector_brusselator_1 * 2
    np.testing.assert_equal(vector_brusselator_res.value, np.ones(2)*2)

def test_vector_brusselator_norm():
    """
    Test norm()
    """
    vector_brusselator = VectorBrusselator()
    vector_brusselator.value = np.array([1, 2])
    np.testing.assert_equal(np.linalg.norm(np.array([1, 2])), vector_brusselator.norm())


def test_vector_brusselator_clone_zero():
    """
    Test clone_zero()
    """
    vector_brusselator = VectorBrusselator()

    vector_brusselator_clone = vector_brusselator.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_brusselator_clone, VectorBrusselator))

    np.testing.assert_equal(vector_brusselator_clone.value, np.zeros(2))


def test_vector_brusselator_clone_rand():
    """
    Test clone_rand()
    """
    vector_brusselator = VectorBrusselator()

    vector_brusselator_clone = vector_brusselator.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_brusselator_clone, VectorBrusselator))


def test_vector_brusselator_set_values():
    """
    Test the set_values()
    """
    vector_brusselator = VectorBrusselator()
    vector_brusselator.set_values(np.array([1, 2]))
    np.testing.assert_equal(vector_brusselator.value, np.array([1, 2]))


def test_vector_brusselator_get_values():
    """
    Test get_values()
    """
    vector_brusselator = VectorBrusselator()
    np.testing.assert_equal(vector_brusselator.get_values(), np.zeros(2))


def test_vector_brusselator_plot_solution():
    """
    Test get_values()
    """
    vector_brusselator = VectorBrusselator()
    np.testing.assert_equal(vector_brusselator.plot_solution(), None)
