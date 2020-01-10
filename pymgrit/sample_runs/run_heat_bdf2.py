import numpy as np
import pathlib

from pymgrit.core import mgrit as solver
from pymgrit.core import grid_transfer_copy
from pymgrit.heat_equation_bdf2 import heat_equation_2pts_bdf2
from pymgrit.heat_equation_bdf2 import heat_equation_2pts_bdf1


def main():
    def output_fcn(self):
        name = 'heat_equation_bdf2'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

    dt = 2 / 128
    heat0 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = heat_equation_2pts_bdf1.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer)
    mgrit.solve()

    heat0 = heat_equation_2pts_bdf2.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = heat_equation_2pts_bdf2.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = heat_equation_2pts_bdf2.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                                 t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, output_fcn=output_fcn)
    return mgrit.solve()


if __name__ == '__main__':
    main()
