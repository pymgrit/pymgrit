import pathlib
import numpy as np

from pymgrit.heat_equation import heat_equation
from pymgrit.core import mgrit as solver
from pymgrit.core import grid_transfer_copy


def main():
    def output_fcn(self):
        name = 'heat_equation'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.solve_iter) + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),
                sol)

    heat0 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=65)
    heat1 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=33)
    heat2 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=17)
    heat3 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=9)
    heat4 = heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, a=1,
                                       t_start=0, t_stop=2, nt=5)

    problem = [heat0, heat1, heat2, heat3, heat4]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, cf_iter=1, cycle_type='F', nested_iteration=False, it=10,
                         output_fcn=output_fcn, output_lvl=2, logging_lvl=20)

    return mgrit.solve()


if __name__ == '__main__':
    main()
