"""
Example 1 from ...
"""

import numpy as np
import matplotlib.pyplot as plt

from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.mgrit_with_plots import MgritWithPlots
from pymgrit.heat.heat_1d_2pts_bdf1 import Heat1DBDF1
from pymgrit.heat.heat_1d_2pts_bdf2 import Heat1DBDF2


def main():
    def rhs(x, t):
        """
        Right-hand side of 1D heat equation example problem at a given space-time point (x,t),
          -sin(pi*x)(sin(t) - a*pi^2*cos(t)),  a = 1

        Note: exact solution is np.sin(np.pi * x) * np.cos(t)
        :param x: spatial grid point
        :param t: time point
        :return: right-hand side of 1D heat equation example problem at point (x,t)
        """

        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def init_cond(x):
        """
        Initial condition of 1D heat equation example,
          u(x,0)  = sin(pi*x)

        :param x: spatial grid point
        :return: initial condition of 1D heat equation example problem
        """
        return np.sin(np.pi * x)

    # Parameters
    t_start = 0
    t_stop = 2
    x_start = 0
    x_end = 1
    nx0 = 2 ** 10 + 1
    a = 1
    nt = 2 ** 10 + 1  # number of time points
    iterations = 1

    # Set up multigrid hierarchy
    heat0 = Heat1D(x_start=x_start, x_end=x_end, nx=nx0, a=a, init_cond=init_cond, rhs=rhs, t_start=t_start,
                   t_stop=t_stop, nt=nt)
    heat1 = Heat1D(x_start=x_start, x_end=x_end, nx=nx0, a=a, init_cond=init_cond, rhs=rhs, t_interval=heat0.t[::4])
    heat2 = Heat1D(x_start=x_start, x_end=x_end, nx=nx0, a=a, init_cond=init_cond, rhs=rhs, t_interval=heat1.t[::4])
    heat3 = Heat1D(x_start=x_start, x_end=x_end, nx=nx0, a=a, init_cond=init_cond, rhs=rhs, t_interval=heat2.t[::4])
    heat4 = Heat1D(x_start=x_start, x_end=x_end, nx=nx0, a=a, init_cond=init_cond, rhs=rhs, t_interval=heat3.t[::4])

    problem = [heat0, heat1, heat2, heat3, heat4]

    # Plots
    mgrit_plots = MgritWithPlots(problem=problem, cf_iter=1, cycle_type='V', random_init_guess=True,
                                 nested_iteration=False)
    mgrit_plots.plot_parallel_distribution(time_procs=4, text_size=13)
    mgrit_plots.plot_cycle(iterations=2, text_size=13)
    mgrit_plots.solve()
    mgrit_plots.plot_convergence(text_size=14)

    # Start timings
    problem = [heat0, heat1, heat2, heat3, heat4]
    runtime = np.zeros(6)
    iters = np.zeros(6)

    # V-cycle, FCF-relaxation
    for i in range(iterations):
        mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='V', random_init_guess=True,
                      nested_iteration=False).solve()
        runtime[1] += mgrit['time_setup'] + mgrit['time_solve']
        iters[1] += len(mgrit['conv'])

    # # V-cycle, FCFCF-relaxation, BDF1
    for i in range(iterations):
        mgrit = Mgrit(problem=problem, cf_iter=2, cycle_type='V', random_init_guess=True,
                      nested_iteration=False).solve()
        runtime[2] += mgrit['time_setup'] + mgrit['time_solve']
        iters[2] += len(mgrit['conv'])

    # F-cycle, F-relaxation
    for i in range(iterations):
        mgrit = Mgrit(problem=problem, cf_iter=0, cycle_type='F', random_init_guess=True,
                      nested_iteration=False).solve()
        runtime[3] += mgrit['time_setup'] + mgrit['time_solve']
        iters[3] += len(mgrit['conv'])

    # F-cycle, FCF-relaxation
    for i in range(iterations):
        mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='F', random_init_guess=True,
                      nested_iteration=False).solve()
        runtime[4] += mgrit['time_setup'] + mgrit['time_solve']
        iters[4] += len(mgrit['conv'])

    #  F-cycle, FCFCF-relaxation
    for i in range(iterations):
        mgrit = Mgrit(problem=problem, cf_iter=2, cycle_type='F', random_init_guess=True,
                      nested_iteration=False).solve()
        runtime[5] += mgrit['time_setup'] + mgrit['time_solve']
        iters[5] += len(mgrit['conv'])

    #Save and print results
    save_runtime = runtime / iterations
    save_iters = iters / iterations
    print('V-cycle with FCF-relaxation:', save_iters[1], save_runtime[1])
    print('V-cycle with FCFCF-relaxation:', save_iters[2], save_runtime[2])
    print('F-cycle with F-relaxation:', save_iters[3], save_runtime[3])
    print('F-cycle with FCF-relaxation:', save_iters[4], save_runtime[4])
    print('F-cycle with FCFCF-relaxation:', save_iters[5], save_runtime[5])
    print(save_iters)


if __name__ == '__main__':
    main()
