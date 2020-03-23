"""
Apply two-level MGRIT with FCF-relaxation to solve the 1D advection equation,
save the MGRIT approximation of the solution after each iteration,
plot the residual history,
plot the initial condition and the MGRIT approximation at the final time
"""

import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from pymgrit.advection.advection_1d import Advection1D
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'advection'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have nx solution values at each time point.
        np.save(path + '/heat_equation_2d-rank' + str(self.comm_time_rank),
                [[[self.t[0][i], self.u[0][i]] for i in self.index_local[0]]])

    # Create two-level time-grid hierarchy for the advection equation
    advection_lvl_0 = Advection1D(c=1, x_start=-1, x_end=1, nx=129, t_start=0, t_stop=2, nt=129)
    advection_lvl_1 = Advection1D(c=1, x_start=-1, x_end=1, nx=129, t_start=0, t_stop=2, nt=65)
    problem = [advection_lvl_0, advection_lvl_1]

    # Set up two-level MGRIT solver with FCF-relaxation without nested iterations
    mgrit = Mgrit(problem=problem, cf_iter=1, nested_iteration=False, output_fcn=output_fcn, output_lvl=2)

    # Solve the advection problem
    info = mgrit.solve()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sol = []
        path = 'results/advection/'
        for filename in os.listdir(path):
            data = np.load(path + filename, allow_pickle=True).tolist()[0]
            sol += data
        sol.sort(key=lambda tup: tup[0])

        # Plot initial condition and MGRIT approximation of the solution at the final time t = 2
        x = advection_lvl_0.x
        fig, ax = plt.subplots()
        ax.plot(x, sol[0][1].get_values(), 'b:', label='initial condition u(x, 0)')
        ax.plot(x, sol[-1][1].get_values(), 'r-', label='solution at final time u(x, 2)')
        ax.legend(loc='lower center', shadow=False, fontsize='x-large')
        plt.xlabel('x')
        plt.show()


if __name__ == '__main__':
    main()
