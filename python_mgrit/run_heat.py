from mgrit import mgrit_fas as solver
from heat_equation import heat_equation
from heat_equation import grid_transfer_copy

if __name__ == '__main__':
    heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=101,
                                       t_start=0, t_stop=2, nt=65)
    heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=101,
                                       t_start=0, t_stop=2, nt=17)
    heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=101,
                                       t_start=0, t_stop=2, nt=5)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer)
    mgrit.solve()
