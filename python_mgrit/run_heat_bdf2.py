from mgrit import mgrit_fas as solver
from heat_equation import heat_equation
from heat_equation import grid_transfer_copy
from heat_equation import heat_equation_2pts_bdf2
from heat_equation import heat_equation_2pts_bdf1
from scipy import linalg as la
import numpy as np

if __name__ == '__main__':
    dt = 2 / 128
    heat0 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer)
    res2 = mgrit.solve()

    heat0 = heat_equation_2pts_bdf2.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001,
                                                 t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer)
    res3 = mgrit.solve()

    exact = problem[0].u_ex
    bdf1 = np.zeros_like(problem[0].u_ex)
    bdf2 = np.zeros_like(problem[0].u_ex)

    j=0
    for i in range(len(res2['u'])):
        bdf1[j] = res2['u'][i].vec_first_time_point
        bdf2[j] = res3['u'][i].vec_first_time_point
        if i != len(res2['u'])-1:
            bdf1[j + 1] = res2['u'][i].vec_second_time_point
            bdf2[j + 1] = res3['u'][i].vec_second_time_point
        j = j + 2

    ebfd1 = exact - bdf1
    ebfd2 = exact - bdf2

    print('bdf1:', la.norm(ebfd1))
    print('bdf1:', la.norm(ebfd2))
