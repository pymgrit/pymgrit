import pathlib
import numpy as np

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():

    def output_fcn(self):
        name = 'scalar_ode'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]]}

        np.save('results/' + name + '/' + str(self.solve_iter) + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),
                sol)

    heat0 = Dahlquist(t_start=0, t_stop=5, nt=11)
    mgrit = Mgrit(problem=simple_setup_problem(problem=heat0, level=2, coarsening=2), it=10, output_fcn=output_fcn)
    return mgrit.solve()




if __name__ == '__main__':
    main()
