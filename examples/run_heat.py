import pathlib
import numpy as np

from pymgrit.heat_equation.heat_equation import HeatEquation
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        name = 'heat_equation'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.solve_iter) + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),
                sol)

    heat0 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65)
    heat1 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=33)
    heat2 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17)
    heat3 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=9)
    heat4 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=5)

    problem = [heat0, heat1, heat2, heat3, heat4]
    mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='F', nested_iteration=False, it=10,
                  output_fcn=output_fcn, output_lvl=2, logging_lvl=20, random_init_guess=False)

    return mgrit.solve()


def main2():
    t_int_0 = np.linspace(0, 2, 65);
    t_int_1 = np.linspace(0, 2, 33);
    t_int_2 = np.linspace(0, 2, 17);
    t_int_3 = np.linspace(0, 2, 9);
    t_int_4 = np.linspace(0, 2, 5);

    heat0 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_interval=t_int_0)
    heat1 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_interval=t_int_1)
    heat2 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_interval=t_int_2)
    heat3 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_interval=t_int_3)
    heat4 = HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_interval=t_int_4)

    problem = [heat0, heat1, heat2, heat3, heat4]
    mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='F', nested_iteration=False, it=10, output_lvl=2,
                  logging_lvl=20, random_init_guess=False)

    return mgrit.solve()


if __name__ == '__main__':
    main()
    main2()
