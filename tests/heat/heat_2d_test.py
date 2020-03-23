"""
Tests heat_2d
"""
import nose
from nose.tools import *
import numpy as np

from pymgrit.heat.heat_2d import Heat2D
from pymgrit.heat.heat_2d import VectorHeat2D


def test_heat_2d_constructor():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny,
                     t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(heat_2d.x_start, x_start)
    np.testing.assert_equal(heat_2d.x_end, x_end)
    np.testing.assert_equal(heat_2d.y_start, y_start)
    np.testing.assert_equal(heat_2d.y_end, y_end)
    np.testing.assert_equal(heat_2d.nx, nx)
    np.testing.assert_equal(heat_2d.ny, ny)
    np.testing.assert_almost_equal(heat_2d.dx, 0.25)
    np.testing.assert_almost_equal(heat_2d.dy, 0.25)
    np.testing.assert_equal(heat_2d.x, np.linspace(x_start, x_end, nx))
    np.testing.assert_equal(heat_2d.y, np.linspace(y_start, y_end, ny))
    np.testing.assert_equal(heat_2d.x_2d, np.linspace(x_start, x_end, nx)[:, np.newaxis])
    np.testing.assert_equal(heat_2d.y_2d, np.linspace(y_start, y_end, ny)[np.newaxis, :])
    np.testing.assert_equal(heat_2d.a, a)

    np.testing.assert_equal(True, isinstance(heat_2d.vector_template, VectorHeat2D))
    np.testing.assert_equal(True, isinstance(heat_2d.vector_t_start, VectorHeat2D))
    np.testing.assert_equal(heat_2d.vector_t_start.get_values(), np.zeros((5, 5)))

    matrix = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0., 0., - 16., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0., 0., - 16., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0., 0., - 16., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0., 0., - 16., 0., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0., 0., - 16., 0., 0., 0.,
               0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., -16., 0., 0., 0., - 16., 0.,
               0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0.,
               0., - 16., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0., 0.,
               0., - 16., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., - 16., 0., 0., 0., - 16., 64., - 16., 0.,
               0., 0., - 16., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    np.testing.assert_equal(heat_2d.space_disc.toarray(), matrix)


def test_heat_2d_constructor_be():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='BE',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)
    np.testing.assert_almost_equal(heat_2d.compute_rhs(u_start=4 * np.ones((5, 5)), t_start=0.2, t_stop=0.3), np.array(
        [0., 0., 0., 0., 0., 0., 4.1625, 4.175, 4.1875, 0., 0., 4.325, 4.35, 4.375, 0., 0., 4.4875, 4.525, 4.5625, 0.,
         0., 0., 0., 0., 0.]))


def test_heat_2d_constructor_fe():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='FE',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)
    np.testing.assert_almost_equal(heat_2d.compute_rhs(u_start=4 * np.ones((5, 5)), t_start=0.2, t_stop=0.3), np.array(
        [0., 0., 0., 0., 0., 0., 4.1625, 4.175, 4.1875, 0., 0., 4.325, 4.35, 4.375, 0., 0., 4.4875, 4.525, 4.5625, 0.,
         0., 0., 0., 0., 0.]))


def test_heat_2d_constructor_cn():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='CN',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)
    np.testing.assert_almost_equal(heat_2d.compute_rhs(u_start=4 * np.ones((5, 5)), t_start=0.2, t_stop=0.3), np.array(
        [0., 0., 0., 0., 0., 0., 4.1625, 4.175, 4.1875, 0., 0., 4.325, 4.35, 4.375, 0., 0., 4.4875, 4.525, 4.5625, 0.,
         0., 0., 0., 0., 0.]))


@raises(Exception)
def test_heat_2d_constructor_exception_method():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='DE',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)


@raises(Exception)
def test_heat_2d_constructor_exception_boundary_left():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, bc_left='2',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)


@raises(Exception)
def test_heat_2d_constructor_exception_boundary_bottom():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, bc_bottom='2',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)


@raises(Exception)
def test_heat_2d_constructor_exception_boundary_right():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, bc_right='2',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)


@raises(Exception)
def test_heat_2d_constructor_exception_boundary_top():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, bc_top='2',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)


def test_heat_2d_constructor_boundary():
    """
    Test constructor
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, bc_top=lambda y: 2,
                     bc_left=lambda y: 2, bc_right=lambda y: 2, bc_bottom=lambda y: 2, rhs=lambda x, y, t: 2 * x * y,
                     t_start=0, t_stop=1, nt=11)


def test_heat_2d_step_be():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='BE',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)

    heat_2d_res = heat_2d.step(u_start=heat_2d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(heat_2d_res.get_values(), np.array(
        [[0., 0., 0., 0., 0.], [0., 0.06659024, 0.08719337, 0.07227713, 0.],
         [0., 0.11922399, 0.15502696, 0.12990086, 0.], [0., 0.12666875, 0.16193148, 0.1391124, 0.],
         [0., 0., 0., 0., 0.]]))


def test_heat_2d_step_cn():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='CN',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)

    heat_2d_res = heat_2d.step(u_start=heat_2d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(heat_2d_res.get_values(), np.array(
        [[0., 0., 0., 0., 0.], [0., 0.09547237, 0.12246323, 0.10480116, 0.],
         [0., 0.17564171, 0.22390841, 0.19336787, 0.], [0., 0.1964882, 0.24654636, 0.21772176, 0.],
         [0., 0., 0., 0., 0.]]))


def test_heat_2d_step_fe():
    """
    Test step()
    """
    x_start = 0
    x_end = 1
    y_start = 3
    y_end = 4
    nx = 5
    ny = 5
    a = 1
    heat_2d = Heat2D(a=a, x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end, nx=nx, ny=ny, method='FE',
                     rhs=lambda x, y, t: 2 * x * y, t_start=0, t_stop=1, nt=11)

    heat_2d_res = heat_2d.step(u_start=heat_2d.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(heat_2d_res.get_values(), np.array(
        [[0., 0., 0., 0., 0.], [0., 0.1625, 0.175, 0.1875, 0.], [0., 0.325, 0.35, 0.375, 0.],
         [0., 0.4875, 0.525, 0.5625, 0.], [0., 0., 0., 0., 0.]]))


def test_vector_heat_2d_constructor():
    """
    Test constructor
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)
    np.testing.assert_equal(vector_heat_2d.nx, 3)
    np.testing.assert_equal(vector_heat_2d.ny, 3)
    np.testing.assert_equal(vector_heat_2d.values, np.zeros((3, 3)))


def test_vector_heat_2d_add():
    """
    Test __add__
    """
    vector_heat_2d_1 = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d_1.values = np.ones((3, 3))
    vector_heat_2d_2 = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d_2.values = 2 * np.ones((3, 3))

    vector_heat_2d_res = vector_heat_2d_1 + vector_heat_2d_2
    np.testing.assert_equal(vector_heat_2d_res.values, 3 * np.ones((3, 3)))


def test_vector_heat_2d_sub():
    """
    Test __sub__
    """
    vector_heat_2d_1 = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d_1.values = np.ones((3, 3))
    vector_heat_2d_2 = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d_2.values = 2 * np.ones((3, 3))

    vector_heat_2d_res = vector_heat_2d_2 - vector_heat_2d_1
    np.testing.assert_equal(vector_heat_2d_res.values, np.ones((3, 3)))


def test_vector_heat_2d_norm():
    """
    Test norm()
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d.values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(np.linalg.norm(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), vector_heat_2d.norm())


def test_vector_heat_2d_clone_zero():
    """
    Test clone_zero()
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)

    vector_heat_2d_clone = vector_heat_2d.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_heat_2d_clone, VectorHeat2D))

    np.testing.assert_equal(vector_heat_2d_clone.values, np.zeros((3, 3)))
    np.testing.assert_equal(vector_heat_2d_clone.nx, 3)
    np.testing.assert_equal(vector_heat_2d_clone.ny, 3)


def test_vector_heat_2d_clone_rand():
    """
    Test clone_rand()
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)

    vector_heat_2d_clone = vector_heat_2d.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_heat_2d_clone, VectorHeat2D))
    np.testing.assert_equal(np.size(vector_heat_2d_clone.values), 9)


def test_vector_heat_2d_set_values():
    """
    Test the set_values()
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)
    vector_heat_2d.set_values(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    np.testing.assert_equal(vector_heat_2d.values, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_vector_heat_2d_get_values():
    """
    Test get_values()
    """
    vector_heat_2d = VectorHeat2D(nx=3, ny=3)
    np.testing.assert_equal(vector_heat_2d.get_values(), np.zeros((3, 3)))


if __name__ == '__main__':
    nose.run()
