import unittest

import numpy as np
from mgrit import mgrit
from mgrit import mgrit as solver
from heat_equation import heat_equation
from heat_equation import grid_transfer_copy


class TestMgritFas(unittest.TestCase):
    def test_split_into(self):
        """
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, a=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        result = np.array([4, 3, 3])
        mgrit = solver.Mgrit(problem=[heat0], transfer=[], nested_iteration=False)
        np.testing.assert_equal(result, mgrit.split_into(10, 3))

    def test_split_points(self):
        """
        """
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, a=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        result = (3, 4)
        mgrit = solver.Mgrit(problem=[heat0], transfer=[], nested_iteration=False)
        np.testing.assert_equal(result, mgrit.split_points(10, 3, 1))

    def test_heat_equation_run(self):
        heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, a=1,
                                           t_start=0, t_stop=2, nt=2 ** 2 + 1)
        heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, a=1,
                                           t_start=0, t_stop=2, nt=2 ** 1 + 1)
        heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=5, a=1,
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

        np.testing.assert_equal(result_t,res['t'])
        np.testing.assert_equal(result_conv, res['conv'])
        np.testing.assert_almost_equal(result_t0_vec, res['u'][0].vec)
        np.testing.assert_almost_equal(result_t1_vec, res['u'][1].vec)
        np.testing.assert_almost_equal(result_t2_vec, res['u'][2].vec)
        np.testing.assert_almost_equal(result_t3_vec, res['u'][3].vec)
        np.testing.assert_almost_equal(result_t4_vec, res['u'][4].vec)


if __name__ == '__main__':
    unittest.main()
