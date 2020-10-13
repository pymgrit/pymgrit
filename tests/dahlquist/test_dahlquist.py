"""
Tests dahlquist
"""
import pytest
import numpy as np

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.dahlquist.dahlquist import VectorDahlquist


def test_dahlquist_constructor():
    """
    Test constructor
    """
    dahlquist = Dahlquist(t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(True, isinstance(dahlquist.vector_template, VectorDahlquist))
    np.testing.assert_equal(True, isinstance(dahlquist.vector_t_start, VectorDahlquist))
    np.testing.assert_equal(dahlquist.vector_t_start.get_values(), 1)
    np.testing.assert_equal('BE', dahlquist.method)


def test_dahlquist_constructor_fe():
    """
    Test constructor
    """
    dahlquist = Dahlquist(t_start=0, t_stop=1, nt=11, method='FE')
    np.testing.assert_equal('FE', dahlquist.method)


def test_dahlquist_constructor_mr():
    """
    Test constructor
    """
    dahlquist = Dahlquist(t_start=0, t_stop=1, nt=11, method='MR')
    np.testing.assert_equal('MR', dahlquist.method)


def test_dahlquist_constructor_tr():
    """
    Test constructor
    """
    dahlquist = Dahlquist(t_start=0, t_stop=1, nt=11, method='TR')
    np.testing.assert_equal('TR', dahlquist.method)


def test_dahlquist_constructor_exception():
    """
    Test constructor
    """
    with pytest.raises(Exception):
        Dahlquist(t_start=0, t_stop=1, nt=11, method='unknown')


def test_dahlquist_step_be():
    """
    Test step()
    """
    dahlquist = Dahlquist(method='BE', t_start=0, t_stop=1, nt=11)
    dahlquist_res = dahlquist.step(u_start=VectorDahlquist(1), t_start=0, t_stop=0.1)
    np.testing.assert_almost_equal(dahlquist_res.get_values(), 0.9090909090909091)


def test_dahlquist_step_fe():
    """
    Test step()
    """
    dahlquist = Dahlquist(method='FE', t_start=0, t_stop=1, nt=11)
    dahlquist_res = dahlquist.step(u_start=VectorDahlquist(1), t_start=0, t_stop=0.1)
    np.testing.assert_almost_equal(dahlquist_res.get_values(), 0.9)


def test_dahlquist_step_tr():
    """
    Test step()
    """
    dahlquist = Dahlquist(method='TR', t_start=0, t_stop=1, nt=11)
    dahlquist_res = dahlquist.step(u_start=VectorDahlquist(1), t_start=0, t_stop=0.1)
    np.testing.assert_almost_equal(dahlquist_res.get_values(), 0.9047619047619047)


def test_dahlquist_step_mr():
    """
    Test step()
    """
    dahlquist = Dahlquist(method='MR', t_start=0, t_stop=1, nt=11)
    dahlquist_res = dahlquist.step(u_start=VectorDahlquist(1), t_start=0, t_stop=0.1)
    np.testing.assert_almost_equal(dahlquist_res.get_values(), 0.9047619047619047)


def test_vector_dahlquist_constructor():
    """
    Test constructor
    """
    vector_dahlquist = VectorDahlquist(1)
    np.testing.assert_equal(vector_dahlquist.value, 1)


def test_vector_dahlquist_add():
    """
    Test __add__
    """
    vector_dahlquist_1 = VectorDahlquist(1)
    vector_dahlquist_2 = VectorDahlquist(2)

    vector_dahlquist_res = vector_dahlquist_1 + vector_dahlquist_2
    np.testing.assert_equal(vector_dahlquist_res.value, 3)


def test_vector_dahlquist_sub():
    """
    Test __sub__
    """
    vector_dahlquist_1 = VectorDahlquist(1)
    vector_dahlquist_2 = VectorDahlquist(2)

    vector_dahlquist_res = vector_dahlquist_2 - vector_dahlquist_1
    np.testing.assert_equal(vector_dahlquist_res.value, 1)


def test_vector_dahlquist_mul():
    """
    Test __sub__
    """
    vector_dahlquist_1 = VectorDahlquist(1)

    vector_dahlquist_res = vector_dahlquist_1 * 7
    np.testing.assert_equal(vector_dahlquist_res.value, 7)


def test_vector_dahlquist_norm():
    """
    Test norm()
    """
    vector_dahlquist = VectorDahlquist(1)
    np.testing.assert_equal(np.linalg.norm(1), vector_dahlquist.norm())


def test_vector_dahlquist_clone_zero():
    """
    Test clone_zero()
    """
    vector_dahlquist = VectorDahlquist(1)

    vector_dahlquist_clone = vector_dahlquist.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_dahlquist_clone, VectorDahlquist))

    np.testing.assert_equal(vector_dahlquist_clone.value, 0)


def test_vector_dahlquist_clone_rand():
    """
    Test clone_rand()
    """
    vector_dahlquist = VectorDahlquist(1)

    vector_dahlquist_clone = vector_dahlquist.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_dahlquist_clone, VectorDahlquist))


def test_vector_dahlquist_set_values():
    """
    Test the set_values()
    """
    vector_dahlquist = VectorDahlquist(1)
    vector_dahlquist.set_values(3)
    np.testing.assert_equal(vector_dahlquist.value, 3)


def test_vector_dahlquist_get_values():
    """
    Test get_values()
    """
    vector_dahlquist = VectorDahlquist(5)
    np.testing.assert_equal(vector_dahlquist.get_values(), 5)
