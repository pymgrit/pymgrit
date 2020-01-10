"""
Tests for the mgrit class
"""
import unittest
import numpy as np

from pymgrit.core import mgrit as solver
from pymgrit.heat_equation import heat_equation
from pymgrit.core import grid_transfer_copy


class TestMgritFas(unittest.TestCase):
    """
    Test class for the Mgrit class
    """

    def test_split_into(self):
        """
        Test the function split_into
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        result = np.array([4, 3, 3])
        mgrit = solver.Mgrit(problem=[heat0], transfer=[], nested_iteration=False)
        np.testing.assert_equal(result, mgrit.split_into(10, 3))

    def test_split_points(self):
        """
        Test the function split points
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        result_proc0 = (4, 0)
        result_proc1 = (3, 4)
        result_proc2 = (3, 7)
        mgrit = solver.Mgrit(problem=[heat0], transfer=[], nested_iteration=False)
        np.testing.assert_equal(result_proc0, mgrit.split_points(10, 3, 0))
        np.testing.assert_equal(result_proc1, mgrit.split_points(10, 3, 1))
        np.testing.assert_equal(result_proc2, mgrit.split_points(10, 3, 2))

    def test_heat_equation_run(self):
        """
        Test one run for the heat equation
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=2 ** 1 + 1)
        heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=2 ** 0 + 1)

        problem = [heat0, heat1, heat2]
        transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
        mgrit = solver.Mgrit(problem=problem, transfer=transfer, cf_iter=1, nested_iteration=True, it=2)
        res = mgrit.solve()

        result_t = np.array([0, 0.5, 1, 1.5, 2])
        result_conv = np.array([0, 0, 0])
        result_t0_vec = np.array([1.0000000e+00, 1.2246468e-16, -1.0000000e+00])
        result_t1_vec = np.array([1.01819672e+00, 1.05735526e-16, -1.01819672e+00])
        result_t2_vec = np.array([6.52749247e-01, 1.58603289e-16, -6.52749247e-01])
        result_t3_vec = np.array([1.00615170e-01, 6.60847038e-17, -1.00615170e-01])
        result_t4_vec = np.array([-4.81527174e-01, -7.93016446e-17, 4.81527174e-01])

        np.testing.assert_equal(result_t, res['t'])
        np.testing.assert_equal(result_conv, res['conv'])
        np.testing.assert_almost_equal(result_t0_vec, res['u'][0].vec)
        np.testing.assert_almost_equal(result_t1_vec, res['u'][1].vec)
        np.testing.assert_almost_equal(result_t2_vec, res['u'][2].vec)
        np.testing.assert_almost_equal(result_t3_vec, res['u'][3].vec)
        np.testing.assert_almost_equal(result_t4_vec, res['u'][4].vec)

    def test_setup_points(self):
        """
        Test for the function setup points
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=65)
        heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=17)
        heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=5)

        problem = [heat0, heat1, heat2]
        transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
        mgrit = solver.Mgrit(problem=problem, transfer=transfer, cf_iter=1, nested_iteration=True, it=2)

        size = 7

        cpts = []
        comm_front = []
        comm_back = []
        block_size_this_lvl = []
        index_local = []
        index_local_c = []
        index_local_f = []
        first_is_c_point = []
        first_is_f_point = []
        last_is_c_point = []
        last_is_f_point = []

        for i in range(size):
            mgrit.comm_time_size = size
            mgrit.comm_time_rank = i

            mgrit.int_start = 0  # First time points of process interval
            mgrit.int_stop = 0  # Last time points of process interval
            mgrit.cpts = []  # C-points per process and level corresponding to complete time interval
            mgrit.comm_front = []  # Communication inside F-relax per MGRIT level
            mgrit.comm_back = []  # Communication inside F-relax per MGRIT level
            mgrit.block_size_this_lvl = []  # Block size per process and level with ghost point
            mgrit.index_local_c = []  # Local indices of C-Points
            mgrit.index_local_f = []  # Local indices of F-Points
            mgrit.index_local = []  # Local indices of all points
            mgrit.first_is_f_point = []  # Communication after C-relax
            mgrit.first_is_c_point = []  # Communication after F-relax
            mgrit.last_is_f_point = []  # Communication after F-relax
            mgrit.last_is_c_point = []  # Communication after C-relax

            for lvl in range(mgrit.lvl_max):
                mgrit.t.append(np.copy(mgrit.problem[lvl].t))
                mgrit.setup_points(lvl=lvl)

            cpts.append(mgrit.cpts)
            comm_front.append(mgrit.comm_front)
            comm_back.append(mgrit.comm_back)
            block_size_this_lvl.append(mgrit.block_size_this_lvl)
            index_local.append(mgrit.index_local)
            index_local_c.append(mgrit.index_local_c)
            index_local_f.append(mgrit.index_local_f)
            first_is_c_point.append(mgrit.first_is_c_point)
            first_is_f_point.append(mgrit.first_is_f_point)
            last_is_c_point.append(mgrit.last_is_c_point)
            last_is_f_point.append(mgrit.last_is_f_point)

        test_cpts = [[np.array([0, 4, 8]), np.array([0]), np.array([0])],
                     [np.array([12, 16]), np.array([4]), np.array([1])],
                     [np.array([20, 24, 28]), np.array([], dtype=int), np.array([], dtype=int)],
                     [np.array([32, 36]), np.array([8]), np.array([2])],
                     [np.array([40, 44]), np.array([], dtype=int), np.array([], dtype=int)],
                     [np.array([48, 52]), np.array([12]), np.array([3])],
                     [np.array([56, 60, 64]), np.array([16]), np.array([4])]]

        test_comm_front = [[False, False, False],
                           [True, True, False],
                           [False, False, False],
                           [False, False, False],
                           [True, True, False],
                           [True, False, False],
                           [False, True, False]]

        test_comm_back = [[True, True, False],
                          [False, False, False],
                          [False, False, False],
                          [True, True, False],
                          [True, False, False],
                          [False, True, False],
                          [False, False, False]]

        test_block_size_this_lvl = [[10, 3, 1],
                                    [11, 3, 2],
                                    [10, 4, 0],
                                    [10, 3, 2],
                                    [10, 3, 0],
                                    [10, 3, 2],
                                    [10, 4, 2]]

        test_index_local = [[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([0, 1, 2]), np.array([0])],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([1, 2]), np.array([1])],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2, 3]), np.array([], dtype=int)],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2]), np.array([1])],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2]), np.array([], dtype=int)],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2]), np.array([1])],
                            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2, 3]), np.array([1])]]

        test_index_local_f = [[np.array([9, 5, 6, 7, 1, 2, 3]), np.array([1, 2]), np.array([], dtype=float)],
                              [np.array([8, 9, 10, 4, 5, 6, 1, 2]), np.array([1]), np.array([], dtype=float)],
                              [np.array([6, 7, 8, 2, 3, 4]), np.array([1, 2, 3]), np.array([], dtype=float)],
                              [np.array([1, 2, 3, 9, 5, 6, 7]), np.array([2]), np.array([], dtype=float)],
                              [np.array([8, 9, 4, 5, 6, 1, 2]), np.array([1, 2]), np.array([], dtype=float)],
                              [np.array([7, 8, 9, 3, 4, 5, 1]), np.array([2]), np.array([], dtype=float)],
                              [np.array([6, 7, 8, 2, 3, 4]), np.array([1, 2]), np.array([], dtype=float)]]

        test_index_local_c = [[np.array([0, 4, 8]), np.array([0]), np.array([0])],
                              [np.array([3, 7]), np.array([2]), np.array([1])],
                              [np.array([1, 5, 9]), np.array([], dtype=int), np.array([], dtype=int)],
                              [np.array([4, 8]), np.array([1]), np.array([1])],
                              [np.array([3, 7]), np.array([], dtype=int), np.array([], dtype=int)],
                              [np.array([2, 6]), np.array([1]), np.array([1])],
                              [np.array([1, 5, 9]), np.array([3]), np.array([1])]]

        test_first_is_c_point = [[False, False, False], [False, False, True], [True, False, False], [False, True, True],
                                 [False, False, False], [False, True, True], [True, False, True]]

        test_first_is_f_point = [[False, False, False], [False, False, False], [False, True, False],
                                 [True, False, False], [False, False, False], [False, False, False],
                                 [False, False, False]]
        test_last_is_f_point = [[False, False, False], [True, False, False], [False, True, False],
                                [False, False, False], [False, True, False], [True, False, False],
                                [False, False, False]]
        test_last_is_c_point = [[False, False, True], [False, True, True], [True, False, False], [False, False, True],
                                [False, False, False], [False, False, True], [False, False, False]]

        for i in range(size):
            assert all([a == b for a, b in zip(first_is_c_point[i], test_first_is_c_point[i])])
            assert all([a == b for a, b in zip(first_is_f_point[i], test_first_is_f_point[i])])
            assert all([a == b for a, b in zip(last_is_f_point[i], test_last_is_f_point[i])])
            assert all([a == b for a, b in zip(last_is_c_point[i], test_last_is_c_point[i])])
            assert all([a == b for a, b in zip(comm_front[i], test_comm_front[i])])
            assert all([a == b for a, b in zip(comm_back[i], test_comm_back[i])])
            assert all([a == b for a, b in zip(block_size_this_lvl[i], test_block_size_this_lvl[i])])
            test = [np.testing.assert_equal(a, b) for a, b in zip(cpts[i], test_cpts[i])]
            test = [np.testing.assert_equal(a, b) for a, b in zip(index_local[i], test_index_local[i])]
            test = [np.testing.assert_equal(a, b) for a, b in zip(index_local_c[i], test_index_local_c[i])]
            test = [np.testing.assert_equal(a, b) for a, b in zip(index_local_f[i], test_index_local_f[i])]

    def test_setup_comm_info(self):
        """
        Test for the function comm_info
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=65)
        heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=17)
        heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, d=1,
                                           t_start=0, t_stop=2, nt=5)

        problem = [heat0, heat1, heat2]
        transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
        mgrit = solver.Mgrit(problem=problem, transfer=transfer, cf_iter=1, nested_iteration=True, it=2)

        size = 7

        send_to = []
        get_from = []

        for i in range(size):
            mgrit.comm_time_size = size
            mgrit.comm_time_rank = i

            mgrit.send_to = []
            mgrit.get_from = []

            mgrit.setup_comm_info()

            send_to.append(mgrit.send_to)
            get_from.append(mgrit.get_from)

        test_send_to = [[1, 1, 1], [2, 2, 3], [3, 3, -99], [4, 4, 5], [5, 5, -99], [6, 6, 6], [-99, -99, -99]]
        test_get_from = [[-99, -99, -99], [0, 0, 0], [1, 1, -99], [2, 2, 1], [3, 3, -99], [4, 4, 3], [5, 5, 5]]

        for i in range(size):
            assert all([a == b for a, b in zip(send_to[i], test_send_to[i])])
            assert all([a == b for a, b in zip(get_from[i], test_get_from[i])])


if __name__ == '__main__':
    unittest.main()
