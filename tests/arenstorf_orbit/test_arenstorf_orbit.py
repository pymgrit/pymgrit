"""
Tests arenstorf_orbit
"""
import numpy as np

from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit
from pymgrit.arenstorf_orbit.arenstorf_orbit import VectorArenstorfOrbit


def test_arenstorf_orbit_constructor():
    """
    Test constructor
    """
    arenstorf_orbit = ArenstorfOrbit(t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(arenstorf_orbit.a, 0.012277471)
    np.testing.assert_equal(arenstorf_orbit.b, 1 - arenstorf_orbit.a)

    np.testing.assert_equal(True, isinstance(arenstorf_orbit.vector_template, VectorArenstorfOrbit))
    np.testing.assert_equal(True, isinstance(arenstorf_orbit.vector_t_start, VectorArenstorfOrbit))
    np.testing.assert_equal(arenstorf_orbit.vector_t_start.get_values(), np.array([0.994, 0.0, 0.0, -2.00158510637908]))


def test_arenstorf_orbit_step():
    """
    Test step()
    """
    arenstorf_orbit = ArenstorfOrbit(t_start=0, t_stop=1, nt=11)
    arenstorf_orbit_res = arenstorf_orbit.step(u_start=VectorArenstorfOrbit(), t_start=0, t_stop=0.1)

    # np.testing.assert_almost_equal(arenstorf_orbit_res.get_values(),
    #                               np.array([5.57160256646951, -0.6298496794201709, 56.043053000678015,
    #                                         -11.976120771048176]))


def test_vector_arenstorf_orbit_constructor():
    """
    Test constructor
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()
    np.testing.assert_equal(vector_arenstorf_orbit.y0, 0)
    np.testing.assert_equal(vector_arenstorf_orbit.y1, 0)
    np.testing.assert_equal(vector_arenstorf_orbit.y2, 0)
    np.testing.assert_equal(vector_arenstorf_orbit.y3, 0)


def test_vector_arenstorf_orbit_add():
    """
    Test __add__
    """
    vector_arenstorf_orbit_1 = VectorArenstorfOrbit()
    vector_arenstorf_orbit_1.y0 = 1
    vector_arenstorf_orbit_1.y1 = 1
    vector_arenstorf_orbit_1.y2 = 1
    vector_arenstorf_orbit_1.y3 = 1
    vector_arenstorf_orbit_2 = VectorArenstorfOrbit()
    vector_arenstorf_orbit_2.y0 = 2
    vector_arenstorf_orbit_2.y1 = 2
    vector_arenstorf_orbit_2.y2 = 2
    vector_arenstorf_orbit_2.y3 = 2

    vector_arenstorf_orbit_res = vector_arenstorf_orbit_1 + vector_arenstorf_orbit_2
    np.testing.assert_equal(vector_arenstorf_orbit_res.y0, 3)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y1, 3)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y2, 3)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y3, 3)


def test_vector_arenstorf_orbit_sub():
    """
    Test __sub__
    """
    vector_arenstorf_orbit_1 = VectorArenstorfOrbit()
    vector_arenstorf_orbit_1.y0 = 1
    vector_arenstorf_orbit_1.y1 = 1
    vector_arenstorf_orbit_1.y2 = 1
    vector_arenstorf_orbit_1.y3 = 1
    vector_arenstorf_orbit_2 = VectorArenstorfOrbit()
    vector_arenstorf_orbit_2.y0 = 2
    vector_arenstorf_orbit_2.y1 = 2
    vector_arenstorf_orbit_2.y2 = 2
    vector_arenstorf_orbit_2.y3 = 2

    vector_arenstorf_orbit_res = vector_arenstorf_orbit_2 - vector_arenstorf_orbit_1
    np.testing.assert_equal(vector_arenstorf_orbit_res.y0, 1)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y1, 1)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y2, 1)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y3, 1)

def test_vector_arenstorf_orbit_mul():
    """
    Test __mul__
    """
    vector_arenstorf_orbit_1 = VectorArenstorfOrbit()
    vector_arenstorf_orbit_1.y0 = 1
    vector_arenstorf_orbit_1.y1 = 1
    vector_arenstorf_orbit_1.y2 = 1
    vector_arenstorf_orbit_1.y3 = 1

    vector_arenstorf_orbit_res = vector_arenstorf_orbit_1 * 5
    np.testing.assert_equal(vector_arenstorf_orbit_res.y0, 5)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y1, 5)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y2, 5)
    np.testing.assert_equal(vector_arenstorf_orbit_res.y3, 5)


def test_vector_arenstorf_orbit_norm():
    """
    Test norm()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()
    vector_arenstorf_orbit.y0 = 1
    vector_arenstorf_orbit.y1 = 2
    vector_arenstorf_orbit.y2 = 3
    vector_arenstorf_orbit.y3 = 4
    np.testing.assert_equal(np.linalg.norm([1, 2, 3, 4]), vector_arenstorf_orbit.norm())


def test_vector_arenstorf_orbit_clone_zero():
    """
    Test clone_zero()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()

    vector_arenstorf_orbit_clone = vector_arenstorf_orbit.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_arenstorf_orbit_clone, VectorArenstorfOrbit))

    np.testing.assert_equal(vector_arenstorf_orbit_clone.y0, 0)
    np.testing.assert_equal(vector_arenstorf_orbit_clone.y1, 0)
    np.testing.assert_equal(vector_arenstorf_orbit_clone.y2, 0)
    np.testing.assert_equal(vector_arenstorf_orbit_clone.y3, 0)


def test_vector_arenstorf_orbit_clone_rand():
    """
    Test clone_rand()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()

    vector_arenstorf_orbit_clone = vector_arenstorf_orbit.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_arenstorf_orbit_clone, VectorArenstorfOrbit))


def test_vector_arenstorf_orbit_set_values():
    """
    Test the set_values()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()
    vector_arenstorf_orbit.set_values(np.array([1, 2, 3, 4]))
    np.testing.assert_equal(vector_arenstorf_orbit.y0, 1)
    np.testing.assert_equal(vector_arenstorf_orbit.y1, 2)
    np.testing.assert_equal(vector_arenstorf_orbit.y2, 3)
    np.testing.assert_equal(vector_arenstorf_orbit.y3, 4)


def test_vector_arenstorf_orbit_get_values():
    """
    Test get_values()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()
    np.testing.assert_equal(vector_arenstorf_orbit.get_values(), np.zeros(4))


def test_vector_arenstorf_orbit_plot():
    """
    Test get_values()
    """
    vector_arenstorf_orbit = VectorArenstorfOrbit()
    np.testing.assert_equal(vector_arenstorf_orbit.plot(), None)
