"""
Apply two-level MGRIT V-cycles with FCF-relaxation to solve the 2D Allen-Cahn equation
    u_t =  (u_xx + u_yy) + 1/ eps^2 u(1-u),  a > 0,
        on [-0.5, 0.5] x [-0.5, 0.5] x (t_start, t_end],
with periodic boundary conditions, eps > 0,
and subject to the initial condition
    u(x,0)  = tanh((R0-|x|)/sqrt(2eps).

Based on https://epubs.siam.org/doi/abs/10.1137/080738398?mobileUi=0 and
https://dl.acm.org/doi/10.1145/3310410
"""

import pathlib
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from mpi4py import MPI

from pymgrit.allen_cahn.allen_cahn import AllenCahn
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/allen_cahn'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution with corresponding time point to file
        np.save(path + '/allen_cahn-rank' + str(self.comm_time_rank),
                [[[self.t[0][i], self.u[0][i]] for i in self.index_local[0]]])


    problem_level_0 = AllenCahn(t_start=0, t_stop=0.032, nt=33, method='IMEX')
    problem_level_1 = AllenCahn(t_interval=problem_level_0.t[::2], method='IMEX')

    # Setup two-level MGRIT solver and solve the problem
    mgrit = Mgrit(problem=[problem_level_0,problem_level_1], output_fcn=output_fcn)
    info = mgrit.solve()

    if MPI.COMM_WORLD.Get_rank() == 0:
        time.sleep(2)
        sol = []
        path = 'results/allen_cahn/'
        for filename in os.listdir(path):
            data = np.load(path + filename, allow_pickle=True).tolist()[0]
            sol += data
        sol.sort(key=lambda tup: tup[0])

        exact_radius = []
        radius = []
        for i in range(len(sol)):
            exact_radius.append(problem_level_0.exact_radius(problem_level_0.t[i]))
            radius.append(problem_level_0.compute_radius(sol[i][1]))

        fig, ax = plt.subplots()
        plt.plot(problem_level_0.t, exact_radius, color='k', linestyle='--', linewidth=1, label='exact')
        plt.plot(problem_level_0.t, radius, linestyle='-', linewidth=2, label='solution')
        ax.set_ylabel('radius')
        ax.set_xlabel('time')
        ax.grid()
        ax.legend(loc=3)
        plt.show()


if __name__ == '__main__':
    main()
