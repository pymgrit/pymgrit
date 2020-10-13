"""
Tests vector_heat_1d_2pts
"""
import numpy as np

from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts


def test_vector_heat_1d_2pts_constructor():
    """
    Test constructor
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=3, dtau=0.1)
    np.testing.assert_equal(vector_heat_1d_2pts.size, 3)
    np.testing.assert_equal(vector_heat_1d_2pts.dtau, 0.1)

    np.testing.assert_equal(vector_heat_1d_2pts.values_first_time_point[0], 0)
    np.testing.assert_equal(vector_heat_1d_2pts.values_first_time_point[1], 0)
    np.testing.assert_equal(vector_heat_1d_2pts.values_first_time_point[2], 0)

    np.testing.assert_equal(vector_heat_1d_2pts.values_second_time_point[0], 0)
    np.testing.assert_equal(vector_heat_1d_2pts.values_second_time_point[1], 0)
    np.testing.assert_equal(vector_heat_1d_2pts.values_second_time_point[2], 0)


def test_vector_heat_1d_2pts_add():
    """
    Test __add__
    """
    vector_heat_1d_2pts_1 = VectorHeat1D2Pts(size=3, dtau=0.1)
    vector_heat_1d_2pts_1.values_first_time_point = np.ones(3)
    vector_heat_1d_2pts_1.values_second_time_point = np.ones(3)
    vector_heat_1d_2pts_2 = VectorHeat1D2Pts(size=3, dtau=0.1)
    vector_heat_1d_2pts_2.values_first_time_point = 2 * np.ones(3)
    vector_heat_1d_2pts_2.values_second_time_point = 2 * np.ones(3)

    vector_heat_1d_2pts_res = vector_heat_1d_2pts_1 + vector_heat_1d_2pts_2
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_first_time_point, 3 * np.ones(3))
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_second_time_point, 3 * np.ones(3))


def test_vector_heat_1d_2pts_sub():
    """
    Test __sub__
    """
    vector_heat_1d_2pts_1 = VectorHeat1D2Pts(size=3, dtau=0.1)
    vector_heat_1d_2pts_1.values_first_time_point = np.ones(3)
    vector_heat_1d_2pts_1.values_second_time_point = np.ones(3)
    vector_heat_1d_2pts_2 = VectorHeat1D2Pts(size=3, dtau=0.1)
    vector_heat_1d_2pts_2.values_first_time_point = 2 * np.ones(3)
    vector_heat_1d_2pts_2.values_second_time_point = 2 * np.ones(3)

    vector_heat_1d_2pts_res = vector_heat_1d_2pts_2 - vector_heat_1d_2pts_1
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_first_time_point, np.ones(3))
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_second_time_point, np.ones(3))

def test_vector_heat_1d_2pts_mul():
    """
    Test __mul__
    """
    vector_heat_1d_2pts_1 = VectorHeat1D2Pts(size=3, dtau=0.1)
    vector_heat_1d_2pts_1.values_first_time_point = np.ones(3)
    vector_heat_1d_2pts_1.values_second_time_point = np.ones(3)

    vector_heat_1d_2pts_res = vector_heat_1d_2pts_1 * 3
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_first_time_point, np.ones(3)*3)
    np.testing.assert_equal(vector_heat_1d_2pts_res.values_second_time_point, np.ones(3)*3)


def test_vector_heat_1d_2pts_norm():
    """
    Test norm()
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=5, dtau=0.1)
    vector_heat_1d_2pts.values_first_time_point = np.array([1, 2, 3, 4, 5])
    vector_heat_1d_2pts.values_second_time_point = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(np.linalg.norm(np.append(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))),
                            vector_heat_1d_2pts.norm())


def test_vector_heat_1d_2pts_clone_zero():
    """
    Test clone_zero()
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=2, dtau=0.1)

    vector_heat_1d_2pts_clone = vector_heat_1d_2pts.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_2pts_clone, VectorHeat1D2Pts))

    np.testing.assert_equal(vector_heat_1d_2pts_clone.values_first_time_point, np.zeros(2))
    np.testing.assert_equal(vector_heat_1d_2pts_clone.values_second_time_point, np.zeros(2))
    np.testing.assert_equal(vector_heat_1d_2pts_clone.size, 2)
    np.testing.assert_equal(vector_heat_1d_2pts_clone.dtau, 0.1)
    np.testing.assert_equal(len(vector_heat_1d_2pts_clone.values_first_time_point), 2)
    np.testing.assert_equal(len(vector_heat_1d_2pts_clone.values_second_time_point), 2)


def test_vector_heat_1d_2pts_clone_rand():
    """
    Test clone_rand()
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=2, dtau=0.1)

    vector_heat_1d_2pts_clone = vector_heat_1d_2pts.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_2pts_clone, VectorHeat1D2Pts))
    np.testing.assert_equal(vector_heat_1d_2pts_clone.size, 2)
    np.testing.assert_equal(vector_heat_1d_2pts_clone.dtau, 0.1)
    np.testing.assert_equal(len(vector_heat_1d_2pts_clone.values_first_time_point), 2)
    np.testing.assert_equal(len(vector_heat_1d_2pts_clone.values_second_time_point), 2)


def test_vector_heat_1d_2pts_set_values():
    """
    Test the set_values()
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=2, dtau=0.1)
    vector_heat_1d_2pts.set_values(first_time_point=np.array([1, 2]), second_time_point=np.array([2, 3]), dtau=0.1)
    np.testing.assert_equal(vector_heat_1d_2pts.values_first_time_point, np.array([1, 2]))
    np.testing.assert_equal(vector_heat_1d_2pts.values_second_time_point, np.array([2, 3]))
    np.testing.assert_equal(vector_heat_1d_2pts.dtau, 0.1)


def test_vector_heat_1d_2pts_get_values():
    """
    Test get_values()
    """
    vector_heat_1d_2pts = VectorHeat1D2Pts(size=5, dtau=0.1)
    np.testing.assert_equal(vector_heat_1d_2pts.get_values()[0], np.zeros(5))
    np.testing.assert_equal(vector_heat_1d_2pts.get_values()[1], np.zeros(5))
    np.testing.assert_equal(vector_heat_1d_2pts.get_values()[2], 0.1)
