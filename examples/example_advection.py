import pathlib
import numpy as np

from pymgrit.advection_1d.advection_1d import Advection1D
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        name = 'advection_equation'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        u_vec = np.zeros((self.problem[0].nt, self.problem[0].nx))
        for i in range(self.problem[0].nt):
            u_vec[i] = self.u[0][i].get_values()
        sol = {'u': [self.u[0][i].get_values() for i in self.index_local[0]], 'u_vec': u_vec, 't': self.problem[0].t,
               'time': self.runtime_solve, 'conv': self.conv}

        np.save('results/' + name + '/' + str(self.solve_iter) + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),
                sol)

    adv0 = Advection1D(c=1, x_start=-16, x_end=16, nx=65, t_start=0, t_stop=6.4, nt=65)
    adv1 = Advection1D(c=1, x_start=-16, x_end=16, nx=65, t_start=0, t_stop=6.4, nt=33)

    # # for time-stepping tests
    # problem = [adv0]
    # transfer = []

    # for 2-level tests
    problem = [adv0, adv1]

    # # F-relax
    # mgrit = solver.Mgrit(problem=problem, cf_iter=0, nested_iteration=False, it=10, tol=1e-50,
    #                         output_fcn=output_fcn, output_lvl=2, logging_lvl=20)

    # FCF-relax
    mgrit = Mgrit(problem=problem, cf_iter=1, nested_iteration=False, it=10, tol=1e-50,
                         output_fcn=output_fcn, output_lvl=2, logging_lvl=20)

    return mgrit.solve()


if __name__ == '__main__':
    main()
