"""
Tests Allen-Cahn
"""
import pytest
import numpy as np

from pymgrit.allen_cahn.allen_cahn import AllenCahn
from pymgrit.allen_cahn.allen_cahn import VectorAllenCahn2D


def test_allen_cahn_constructor():
    """
    Test constructor
    """
    nx = 3
    nu = 1
    eps = 3
    newton_maxiter = 4
    newton_tol = 5
    lin_tol = 5
    lin_maxiter = 1
    radius = 0.25
    method = 'CN'
    allen_cahn = AllenCahn(nx=nx, eps=eps, newton_tol=newton_tol, newton_maxiter=newton_maxiter, lin_tol=lin_tol,
                           lin_maxiter=lin_maxiter, radius=radius, method=method, nu=nu, t_start=0, t_stop=1, nt=11)

    np.testing.assert_equal(allen_cahn.nx, nx)
    np.testing.assert_equal(allen_cahn.nu, nu)
    np.testing.assert_equal(allen_cahn.eps, eps)
    np.testing.assert_equal(allen_cahn.newton_maxiter, newton_maxiter)
    np.testing.assert_equal(allen_cahn.newton_tol, newton_tol)
    np.testing.assert_equal(allen_cahn.lin_tol, lin_tol)
    np.testing.assert_equal(allen_cahn.lin_maxiter, lin_maxiter)
    np.testing.assert_equal(allen_cahn.radius, radius)
    np.testing.assert_equal(allen_cahn.method, method)

    np.testing.assert_equal(True, isinstance(allen_cahn.vector_template, VectorAllenCahn2D))
    np.testing.assert_equal(True, isinstance(allen_cahn.vector_t_start, VectorAllenCahn2D))
    np.testing.assert_almost_equal(allen_cahn.vector_t_start.get_values(),
                                   np.array([[-0.10732614, -0.05885746, -0.10732614],
                                             [-0.05885746, 0.05885746, -0.05885746],
                                             [-0.10732614, -0.05885746, -0.10732614]]))

    matrix = np.array([[-36., 9., 9., 9., 0., 0., 9., 0., 0.],
                       [9., -36., 9., 0., 9., 0., 0., 9., 0.],
                       [9., 9., -36., 0., 0., 9., 0., 0., 9.],
                       [9., 0., 0., -36., 9., 9., 9., 0., 0.],
                       [0., 9., 0., 9., -36., 9., 0., 9., 0.],
                       [0., 0., 9., 9., 9., - 36., 0., 0., 9.],
                       [9., 0., 0., 9., 0., 0., - 36., 9., 9.],
                       [0., 9., 0., 0., 9., 0., 9., - 36., 9.],
                       [0., 0., 9., 0., 0., 9., 9., 9., - 36.]])
    np.testing.assert_equal(allen_cahn.space_disc.toarray(), matrix)


def test_allen_cahn_constructor_exception_method():
    """
    Test constructor
    """
    nx = 3
    nu = 1
    eps = 3
    newton_maxiter = 4
    newton_tol = 5
    lin_tol = 5
    lin_maxiter = 1
    radius = 0.25
    method = 'DE'
    with pytest.raises(Exception):
        allen_cahn = AllenCahn(nx=nx, eps=eps, newton_tol=newton_tol, newton_maxiter=newton_maxiter, lin_tol=lin_tol,
                               lin_maxiter=lin_maxiter, radius=radius, method=method, nu=nu, t_start=0, t_stop=1, nt=11)


def test_allen_cahn_step_cn():
    """
    Test step()
    """
    nx = 3
    nu = 1
    eps = 3
    newton_maxiter = 4
    newton_tol = 5
    lin_tol = 5
    lin_maxiter = 1
    radius = 0.25
    method = 'CN'
    allen_cahn = AllenCahn(nx=nx, eps=eps, newton_tol=newton_tol, newton_maxiter=newton_maxiter, lin_tol=lin_tol,
                           lin_maxiter=lin_maxiter, radius=radius, method=method, nu=nu, t_start=0, t_stop=1, nt=11)

    allen_cahn_res = allen_cahn.step(u_start=allen_cahn.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(allen_cahn_res.get_values(),
                                   np.array([[-0.10732614, -0.05885746, -0.10732614],
                                             [-0.05885746, 0.05885746, -0.05885746],
                                             [-0.10732614, -0.05885746, -0.10732614]]))


def test_allen_cahn_step_impl():
    """
    Test step()
    """
    nx = 3
    nu = 1
    eps = 3
    newton_maxiter = 4
    newton_tol = 5
    lin_tol = 5
    lin_maxiter = 1
    radius = 0.25
    method = 'IMPL'
    allen_cahn = AllenCahn(nx=nx, eps=eps, newton_tol=newton_tol, newton_maxiter=newton_maxiter, lin_tol=lin_tol,
                           lin_maxiter=lin_maxiter, radius=radius, method=method, nu=nu, t_start=0, t_stop=1, nt=11)

    allen_cahn_res = allen_cahn.step(u_start=allen_cahn.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(allen_cahn_res.get_values(),
                                   np.array([[-0.10732614, -0.05885746, -0.10732614],
                                             [-0.05885746, 0.05885746, -0.05885746],
                                             [-0.10732614, -0.05885746, -0.10732614]]))


def test_heat_2d_step_imex():
    """
    Test step()
    """
    nx = 3
    nu = 1
    eps = 3
    newton_maxiter = 4
    newton_tol = 5
    lin_tol = 5
    lin_maxiter = 1
    radius = 0.25
    method = 'IMEX'
    allen_cahn = AllenCahn(nx=nx, eps=eps, newton_tol=newton_tol, newton_maxiter=newton_maxiter, lin_tol=lin_tol,
                           lin_maxiter=lin_maxiter, radius=radius, method=method, nu=nu, t_start=0, t_stop=1, nt=11)

    allen_cahn_res = allen_cahn.step(u_start=allen_cahn.vector_t_start, t_start=0, t_stop=0.1)

    np.testing.assert_almost_equal(allen_cahn_res.get_values(),
                                   np.array([[-0.07997795, -0.0640509, -0.07997795],
                                             [-0.0640509, -0.03719789, -0.0640509],
                                             [-0.07997795, -0.0640509, -0.07997795]]))


def test_vector_allen_cahn_constructor():
    """
    Test constructor
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)
    np.testing.assert_equal(vector.nx, 3)
    np.testing.assert_equal(vector.ny, 3)
    np.testing.assert_equal(vector.values, np.zeros((3, 3)))


def test_vector_allen_cahn_add():
    """
    Test __add__
    """
    vector_1 = VectorAllenCahn2D(nx=3, ny=3)
    vector_1.values = np.ones((3, 3))
    vector_2 = VectorAllenCahn2D(nx=3, ny=3)
    vector_2.values = 2 * np.ones((3, 3))

    vector_heat_2d_res = vector_1 + vector_2
    np.testing.assert_equal(vector_heat_2d_res.values, 3 * np.ones((3, 3)))


def test_vector_allen_cahn_sub():
    """
    Test __sub__
    """
    vector_1 = VectorAllenCahn2D(nx=3, ny=3)
    vector_1.values = np.ones((3, 3))
    vector_2 = VectorAllenCahn2D(nx=3, ny=3)
    vector_2.values = 2 * np.ones((3, 3))

    vector_heat_2d_res = vector_2 - vector_1
    np.testing.assert_equal(vector_heat_2d_res.values, np.ones((3, 3)))


def test_vector_allen_cahn_mul():
    """
    Test __mul__
    """
    vector_1 = VectorAllenCahn2D(nx=3, ny=3)
    vector_1.values = np.ones((3, 3))

    vector_heat_2d_res = vector_1 * 5
    np.testing.assert_equal(vector_heat_2d_res.values, 5 * np.ones((3, 3)))


def test_vector_allen_cahn_norm():
    """
    Test norm()
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)
    vector.values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(np.linalg.norm(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), vector.norm())


def test_vector_allen_cahn_clone_zero():
    """
    Test clone_zero()
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)

    vector_clone = vector.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_clone, VectorAllenCahn2D))

    np.testing.assert_equal(vector_clone.values, np.zeros((3, 3)))
    np.testing.assert_equal(vector_clone.nx, 3)
    np.testing.assert_equal(vector_clone.ny, 3)


def test_vector_allen_cahn_clone_rand():
    """
    Test clone_rand()
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)

    vector_clone = vector.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_clone, VectorAllenCahn2D))
    np.testing.assert_equal(np.size(vector_clone.values), 9)


def test_vector_allen_cahn_set_values():
    """
    Test the set_values()
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)
    vector.set_values(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    np.testing.assert_equal(vector.values, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_vector_allen_cahn_get_values():
    """
    Test get_values()
    """
    vector = VectorAllenCahn2D(nx=3, ny=3)
    np.testing.assert_equal(vector.get_values(), np.zeros((3, 3)))
