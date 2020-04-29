"""
Tests heat_1d_2pts_bdf1
"""
import numpy as np

from pymgrit.heat.heat_1d_2pts_bdf1 import Heat1DBDF1
from pymgrit.heat.vector_heat_1d_2pts import VectorHeat1D2Pts


def test_heat_1d_2pts_bdf1_constructor():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    nx = 11
    a = 1
    dtau = 0.1
    heat_1d = Heat1DBDF1(a=a, x_start=x_start, x_end=x_end, nx=nx, dtau=dtau, t_start=0, t_stop=1, nt=11)

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


def test_heat_1d_2pts_bdf1_step():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    nx = 11
    a = 1
    dtau = 0.1
    heat_1d = Heat1DBDF1(a=a, init_cond=lambda x: 2 * x, x_start=x_start, x_end=x_end, nx=nx, dtau=dtau, t_start=0,
                         t_stop=1, nt=11)
    heat_1d_res = heat_1d.step(u_start=heat_1d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_equal(True, isinstance(heat_1d_res, VectorHeat1D2Pts))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[0], np.array(
        [0.14498001, 0.28445802, 0.41238183, 0.52154382, 0.6028602, 0.6444626, 0.63051125, 0.53961104, 0.34267192]))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[1], np.array(
        [0.08691756, 0.16802887, 0.23749726, 0.2894772, 0.31825048, 0.31856279, 0.28628511, 0.21958482, 0.12088191]))
    np.testing.assert_almost_equal(heat_1d_res.get_values()[2], 0.1)
