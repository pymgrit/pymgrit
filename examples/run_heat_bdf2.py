import numpy as np
import pathlib

from pymgrit.core.mgrit import Mgrit
from pymgrit.heat_equation_bdf2.heat_equation_2pts_bdf1 import HeatEquationBDF1
from pymgrit.heat_equation_bdf2.heat_equation_2pts_bdf2 import HeatEquationBDF2


def main():
    def output_fcn(self):
        name = 'heat_equation_bdf2'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

    dt = 2 / 128
    heat0 = HeatEquationBDF1(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = HeatEquationBDF1(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = HeatEquationBDF1(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    mgrit.solve()

    heat0 = HeatEquationBDF2(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65, dt=dt)
    heat1 = HeatEquationBDF2(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17, dt=dt)
    heat2 = HeatEquationBDF2(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=5, dt=dt)

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem, output_fcn=output_fcn)
    return mgrit.solve()


if __name__ == '__main__':
    main()
