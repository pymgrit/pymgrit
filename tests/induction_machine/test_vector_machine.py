"""
Tests vector_heat_1d_2pts
"""
import numpy as np

from pymgrit.induction_machine.vector_machine import VectorMachine


def test_vector_machine_constructor():
    """
    Test constructor
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    np.testing.assert_equal(vector_machine.u_front_size, 4)
    np.testing.assert_equal(vector_machine.u_middle_size, 5)
    np.testing.assert_equal(vector_machine.u_back_size, 6)

    np.testing.assert_equal(vector_machine.u_front, np.zeros(4))
    np.testing.assert_equal(vector_machine.u_middle, np.zeros(5))
    np.testing.assert_equal(vector_machine.u_back, np.zeros(6))

    np.testing.assert_equal(vector_machine.jl, 0)
    np.testing.assert_equal(vector_machine.ua, 0)
    np.testing.assert_equal(vector_machine.ub, 0)
    np.testing.assert_equal(vector_machine.uc, 0)
    np.testing.assert_equal(vector_machine.ia, 0)
    np.testing.assert_equal(vector_machine.ib, 0)
    np.testing.assert_equal(vector_machine.ic, 0)
    np.testing.assert_equal(vector_machine.tr, 0)


def test_vector_machine_add():
    """
    Test __add__
    """
    vector_machine_1 = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine_1.u_front = np.ones(4)
    vector_machine_1.u_middle = np.ones(5)
    vector_machine_1.u_back = np.ones(6)
    vector_machine_1.ua = 1
    vector_machine_1.ub = 2
    vector_machine_1.uc = 3
    vector_machine_1.ia = 4
    vector_machine_1.ib = 5
    vector_machine_1.ic = 6
    vector_machine_1.tr = 7
    vector_machine_1.jl = 8
    vector_machine_2 = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine_2.u_front = 2 * np.ones(4)
    vector_machine_2.u_middle = 3 * np.ones(5)
    vector_machine_2.u_back = 4 * np.ones(6)
    vector_machine_2.ua = 11
    vector_machine_2.ub = 12
    vector_machine_2.uc = 13
    vector_machine_2.ia = 14
    vector_machine_2.ib = 15
    vector_machine_2.ic = 16
    vector_machine_2.tr = 17
    vector_machine_2.jl = 18

    vector_machine_res = vector_machine_1 + vector_machine_2
    np.testing.assert_equal(vector_machine_res.u_front, 3 * np.ones(4))
    np.testing.assert_equal(vector_machine_res.u_middle, 4 * np.ones(5))
    np.testing.assert_equal(vector_machine_res.u_back, 5 * np.ones(6))

    np.testing.assert_equal(vector_machine_res.ua, 12)
    np.testing.assert_equal(vector_machine_res.ub, 14)
    np.testing.assert_equal(vector_machine_res.uc, 16)
    np.testing.assert_equal(vector_machine_res.ia, 18)
    np.testing.assert_equal(vector_machine_res.ib, 20)
    np.testing.assert_equal(vector_machine_res.ic, 22)
    np.testing.assert_equal(vector_machine_res.tr, 24)
    np.testing.assert_equal(vector_machine_res.jl, 26)


def test_vector_machine_sub():
    """
    Test __sub__
    """
    vector_machine_1 = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine_1.u_front = np.ones(4)
    vector_machine_1.u_middle = np.ones(5)
    vector_machine_1.u_back = np.ones(6)
    vector_machine_1.ua = 1
    vector_machine_1.ub = 2
    vector_machine_1.uc = 3
    vector_machine_1.ia = 4
    vector_machine_1.ib = 5
    vector_machine_1.ic = 6
    vector_machine_1.tr = 7
    vector_machine_1.jl = 8
    vector_machine_2 = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine_2.u_front = 2 * np.ones(4)
    vector_machine_2.u_middle = 3 * np.ones(5)
    vector_machine_2.u_back = 4 * np.ones(6)
    vector_machine_2.ua = 11
    vector_machine_2.ub = 12
    vector_machine_2.uc = 13
    vector_machine_2.ia = 14
    vector_machine_2.ib = 15
    vector_machine_2.ic = 16
    vector_machine_2.tr = 17
    vector_machine_2.jl = 18

    vector_machine_res = vector_machine_2 - vector_machine_1
    np.testing.assert_equal(vector_machine_res.u_front, np.ones(4))
    np.testing.assert_equal(vector_machine_res.u_middle, 2 * np.ones(5))
    np.testing.assert_equal(vector_machine_res.u_back, 3 * np.ones(6))

    np.testing.assert_equal(vector_machine_res.jl, 10)
    np.testing.assert_equal(vector_machine_res.ua, 10)
    np.testing.assert_equal(vector_machine_res.ub, 10)
    np.testing.assert_equal(vector_machine_res.uc, 10)
    np.testing.assert_equal(vector_machine_res.ia, 10)
    np.testing.assert_equal(vector_machine_res.ib, 10)
    np.testing.assert_equal(vector_machine_res.ic, 10)
    np.testing.assert_equal(vector_machine_res.tr, 10)

def test_vector_machine_mul():
    """
    Test __mul__
    """
    vector_machine_1 = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine_1.u_front = np.ones(4)
    vector_machine_1.u_middle = np.ones(5)
    vector_machine_1.u_back = np.ones(6)
    vector_machine_1.ua = 1
    vector_machine_1.ub = 2
    vector_machine_1.uc = 3
    vector_machine_1.ia = 4
    vector_machine_1.ib = 5
    vector_machine_1.ic = 6
    vector_machine_1.tr = 7
    vector_machine_1.jl = 8

    fac = 5
    vector_machine_res = vector_machine_1 * fac
    np.testing.assert_equal(vector_machine_res.u_front, np.ones(4)*fac)
    np.testing.assert_equal(vector_machine_res.u_middle, fac * np.ones(5))
    np.testing.assert_equal(vector_machine_res.u_back, fac * np.ones(6))

    np.testing.assert_equal(vector_machine_res.jl, 8*fac)
    np.testing.assert_equal(vector_machine_res.ua, 1*fac)
    np.testing.assert_equal(vector_machine_res.ub, 2*fac)
    np.testing.assert_equal(vector_machine_res.uc, 3*fac)
    np.testing.assert_equal(vector_machine_res.ia, 4*fac)
    np.testing.assert_equal(vector_machine_res.ib, 5*fac)
    np.testing.assert_equal(vector_machine_res.ic, 6*fac)
    np.testing.assert_equal(vector_machine_res.tr, 7*fac)

def test_vector_machine_norm():
    """
    Test norm()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine.u_front = 2 * np.ones(4)
    vector_machine.u_middle = 3 * np.ones(5)
    vector_machine.u_back = 4 * np.ones(6)
    vector_machine.ua = 11
    vector_machine.ub = 12
    vector_machine.uc = 13
    vector_machine.ia = 14
    vector_machine.ib = 15
    vector_machine.ic = 16
    vector_machine.tr = 17
    vector_machine.jl = 18
    np.testing.assert_equal(np.linalg.norm(np.array([2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])),
                            vector_machine.norm())


def test_vector_machine_clone_zero():
    """
    Test clone_zero()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)

    vector_heat_1d_2pts_clone = vector_machine.clone_zero()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_2pts_clone, VectorMachine))

    np.testing.assert_equal(vector_machine.u_front_size, 4)
    np.testing.assert_equal(vector_machine.u_middle_size, 5)
    np.testing.assert_equal(vector_machine.u_back_size, 6)

    np.testing.assert_equal(vector_machine.u_front, np.zeros(4))
    np.testing.assert_equal(vector_machine.u_middle, np.zeros(5))
    np.testing.assert_equal(vector_machine.u_back, np.zeros(6))

    np.testing.assert_equal(vector_machine.jl, 0)
    np.testing.assert_equal(vector_machine.ua, 0)
    np.testing.assert_equal(vector_machine.ub, 0)
    np.testing.assert_equal(vector_machine.uc, 0)
    np.testing.assert_equal(vector_machine.ia, 0)
    np.testing.assert_equal(vector_machine.ib, 0)
    np.testing.assert_equal(vector_machine.ic, 0)
    np.testing.assert_equal(vector_machine.tr, 0)


def test_vector_vector_machine_cole_rand():
    """
    Test clone_rand()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)

    vector_heat_1d_2pts_clone = vector_machine.clone_rand()

    np.testing.assert_equal(True, isinstance(vector_heat_1d_2pts_clone, VectorMachine))

    np.testing.assert_equal(vector_machine.u_front_size, 4)
    np.testing.assert_equal(vector_machine.u_middle_size, 5)
    np.testing.assert_equal(vector_machine.u_back_size, 6)

    np.testing.assert_equal(vector_machine.u_front, np.zeros(4))
    np.testing.assert_equal(vector_machine.u_middle, np.zeros(5))
    np.testing.assert_equal(vector_machine.u_back, np.zeros(6))

    np.testing.assert_equal(vector_machine.jl, 0)
    np.testing.assert_equal(vector_machine.ua, 0)
    np.testing.assert_equal(vector_machine.ub, 0)
    np.testing.assert_equal(vector_machine.uc, 0)
    np.testing.assert_equal(vector_machine.ia, 0)
    np.testing.assert_equal(vector_machine.ib, 0)
    np.testing.assert_equal(vector_machine.ic, 0)
    np.testing.assert_equal(vector_machine.tr, 0)


def test_vector_machine_set_values():
    """
    Test the set_values()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine.set_values(values=np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]),
                              jl=1,
                              ia=2,
                              ib=3,
                              ic=4,
                              ua=5,
                              ub=6,
                              uc=7,
                              tr=8)

    np.testing.assert_equal(vector_machine.u_front_size, 4)
    np.testing.assert_equal(vector_machine.u_middle_size, 5)
    np.testing.assert_equal(vector_machine.u_back_size, 6)

    np.testing.assert_equal(vector_machine.u_front, np.ones(4))
    np.testing.assert_equal(vector_machine.u_middle, 2 * np.ones(5))
    np.testing.assert_equal(vector_machine.u_back, 3 * np.ones(6))

    np.testing.assert_equal(vector_machine.jl, 1)
    np.testing.assert_equal(vector_machine.ia, 2)
    np.testing.assert_equal(vector_machine.ib, 3)
    np.testing.assert_equal(vector_machine.ic, 4)
    np.testing.assert_equal(vector_machine.ua, 5)
    np.testing.assert_equal(vector_machine.ub, 6)
    np.testing.assert_equal(vector_machine.uc, 7)
    np.testing.assert_equal(vector_machine.tr, 8)


def test_vector_vector_machine_get_values():
    """
    Test get_values()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    np.testing.assert_equal(vector_machine.get_values(), np.zeros(15))


def test_vector_vector_machine_pack():
    """
    Test get_values()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)
    vector_machine.set_values(values=np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]),
                              jl=1,
                              ia=2,
                              ib=3,
                              ic=4,
                              ua=5,
                              ub=6,
                              uc=7,
                              tr=8)
    np.testing.assert_equal(vector_machine.pack(),
                            [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2, 2]), np.array([3, 3, 3, 3, 3, 3]), 1, 2, 3,
                             4, 5, 6, 7, 8])
    import pickle
    try:
        pickle.dumps(vector_machine.pack())
    except pickle.PicklingError:
        pickle_test = False
    pickle_test = True
    np.testing.assert_equal(pickle_test, True)


def test_vector_vector_machine_unpack():
    """
    Test get_values()
    """
    vector_machine = VectorMachine(u_front_size=4, u_middle_size=5, u_back_size=6)

    vector_machine.unpack(
        values=[np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2, 2]), np.array([3, 3, 3, 3, 3, 3]), 1, 2, 3,
                4, 5, 6, 7, 8])

    np.testing.assert_equal(vector_machine.u_front_size, 4)
    np.testing.assert_equal(vector_machine.u_middle_size, 5)
    np.testing.assert_equal(vector_machine.u_back_size, 6)

    np.testing.assert_equal(vector_machine.u_front, np.ones(4))
    np.testing.assert_equal(vector_machine.u_middle, 2 * np.ones(5))
    np.testing.assert_equal(vector_machine.u_back, 3 * np.ones(6))

    np.testing.assert_equal(vector_machine.jl, 1)
    np.testing.assert_equal(vector_machine.ia, 2)
    np.testing.assert_equal(vector_machine.ib, 3)
    np.testing.assert_equal(vector_machine.ic, 4)
    np.testing.assert_equal(vector_machine.ua, 5)
    np.testing.assert_equal(vector_machine.ub, 6)
    np.testing.assert_equal(vector_machine.uc, 7)
    np.testing.assert_equal(vector_machine.tr, 8)
