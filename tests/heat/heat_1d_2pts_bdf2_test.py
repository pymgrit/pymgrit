"""
Tests heat_1d_2pts_bdf2
"""
import nose
import numpy as np

from pymgrit.heat.heat_1d_2pts_bdf2 import Heat1DBDF2
from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts


def test_heat_1d_2pts_bdf2_constructor():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    nx = 11
    a = 1
    dtau = 0.1
    heat_1d = Heat1DBDF2(a=a, x_start=x_start, x_end=x_end, nx=nx, dtau=dtau, t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(heat_1d.x_start, x_start)
    np.testing.assert_equal(heat_1d.x_end, x_end)
    np.testing.assert_equal(heat_1d.nx, nx - 2)
    np.testing.assert_almost_equal(heat_1d.dx, 0.1)
    np.testing.assert_almost_equal(heat_1d.a, a)
    np.testing.assert_equal(heat_1d.x, np.linspace(x_start, x_end, nx)[1:-1])

    np.testing.assert_equal(True, isinstance(heat_1d.vector_template, VectorHeat1D2Pts))
    np.testing.assert_equal(True, isinstance(heat_1d.vector_t_start, VectorHeat1D2Pts))
    np.testing.assert_equal(heat_1d.vector_t_start.get_values()[0], np.zeros(9))
    np.testing.assert_equal(heat_1d.vector_t_start.get_values()[1], np.zeros(9))
    np.testing.assert_equal(heat_1d.vector_t_start.get_values()[2], 0.1)


def test_heat_1d_2pts_bdf2_step():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    nx = 11
    a = 1
    dtau = 0.1
    heat_1d = Heat1DBDF2(a=a, init_cond=lambda x: 2 * x, x_start=x_start, x_end=x_end, nx=nx, dtau=dtau, t_start=0,
                         t_stop=1, nt=5)
    np.testing.assert_almost_equal(heat_1d.vector_t_start.get_values()[0],
                                   np.array([0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8]))
    np.testing.assert_almost_equal(heat_1d.vector_t_start.get_values()[1], np.array(
        [0.15656217, 0.30443677, 0.43319873, 0.52860043, 0.56972221, 0.52478844, 0.34481236, -0.04620125, -0.76645512]))
    np.testing.assert_almost_equal(heat_1d.vector_t_start.get_values()[2], 0.1)
    heat_1d_res = heat_1d.step(u_start=heat_1d.vector_t_start, t_start=0, t_stop=0.2)

    np.testing.assert_equal(True, isinstance(heat_1d_res, VectorHeat1D2Pts))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[0], np.array(
        [0.07115547, 0.13167183, 0.17105162, 0.1794494, 0.1490445, 0.07705183, -0.02834074, -0.1369469, -0.17685485]))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[1], np.array(
        [0.01235156, 0.02015287, 0.01986458, 0.01000559, -0.00781242, -0.02812508, -0.04182745, -0.03889518,
         -0.01671786]))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[2], 0.1)


if __name__ == '__main__':
    nose.run()
