"""
Apply two-level MGRIT with FCF-relaxation
to solve the 1D advection equation
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.advection.advection_1d import Advection1D
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'advection' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have nx solution values at each time point.
        np.save(path + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),  # Local time interval
                [self.u[0][i] for i in self.index_local[0]])                # Solution values at local time points

    # Create two-level time-grid hierarchy for the advection equation
    advection_lvl_0 = Advection1D(c=1, x_start=-1, x_end=1, nx=129, t_start=0, t_stop=2, nt=129)
    advection_lvl_1 = Advection1D(c=1, x_start=-1, x_end=1, nx=129, t_start=0, t_stop=2, nt=65)

    # # For time-stepping tests
    # problem = [advection_lvl_0]

    # For two-level tests
    problem = [advection_lvl_0, advection_lvl_1]

    # Set up two-level MGRIT solver with FCF-relaxation
    mgrit = Mgrit(problem=problem, cf_iter=1, nested_iteration=False, output_fcn=output_fcn, output_lvl=2)

    info = mgrit.solve()

    # plot residual history
    plt.figure(1)
    res = info['conv']
    iters = np.arange(1, res.size + 1)
    plt.semilogy(iters, res, 'o-')
    plt.xticks(iters)
    plt.xlabel('iter #')
    plt.ylabel('residual norm')

    # plot initial condition and solution at final time t = 2
    sol = np.load('results/advection/' + str(len(res)) + '/0.0:2.0.npy', allow_pickle=True)
    x = advection_lvl_0.x
    fig, ax = plt.subplots()
    ax.plot(x, sol[0].get_values(), 'b:', label='initial condition u(x, 0)')
    ax.plot(x, sol[-1].get_values(), 'r-', label='solution at final time u(x, 2)')
    ax.legend(loc='lower center', shadow=False, fontsize='x-large')
    plt.xlabel('x')
    plt.show()


if __name__ == '__main__':
    main()
