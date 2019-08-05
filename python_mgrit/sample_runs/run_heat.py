from mgrit import mgrit as solver
from heat_equation import heat_equation
from heat_equation import grid_transfer_copy
import pathlib
import numpy as np

if __name__ == '__main__':
    def output_fcn(self):
        name = 'heat_equation'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)


    heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=2 ** 5 + 1)
    heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=2 ** 4 + 1)
    heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=2 ** 3 + 1)
    heat3 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=2 ** 2 + 1)
    heat4 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=2 ** 1 + 1)

    problem = [heat0, heat1, heat2, heat3, heat4]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, cf_iter=1, nested_iteration=True, it=5,
                         output_fcn=output_fcn)
    res = mgrit.solve()
