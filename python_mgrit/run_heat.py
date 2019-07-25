from mgrit import mgrit_fas as solver
from heat_equation import heat_equation
from heat_equation import grid_transfer_copy

if __name__ == '__main__':
    heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001,
                                       t_start=0, t_stop=2, nt=2**5+1)
    heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001,
                                       t_start=0, t_stop=2, nt=2**4+1)
    heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001,
                                       t_start=0, t_stop=2, nt=2**3+1)
    heat3 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001,
                                       t_start=0, t_stop=2, nt=2**2+1)
    heat4 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001,
                                       t_start=0, t_stop=2, nt=2**1+1)

    problem = [heat0, heat1, heat2, heat3, heat4]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer, cf_iter=1, nested_iteration=True, it=5)
    res = mgrit.solve()

    #print(res['u'])