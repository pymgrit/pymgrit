"""
Example 1 from ...
"""

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

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

    def exact_sol(x, t):
        return np.sin(np.pi * x) * np.cos(t)

    def plot_error(mgrit, ts):
        fonts = 18
        lw = 2
        ms = 10
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        fig1 = plt.figure(figsize=(15, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        labels = [
            'V-cycle, FCF',
            'V-cycle, FCFCF',
            'F-cycle, F',
            'F-cycle, FCF',
            'F-cycle, FCFCF',
        ]
        colors = [
            'green',
            'black',
            'orange',
            'yellow',
            'purple'
        ]
        mfc = ['green', 'black', 'white', 'white', 'white']
        marker = ['s', 'D', 'o', 's', 'D']
        linetype = ['-', '-', '--', '--', '--']
        count = 1
        save_vecs = []
        for j in range(len(mgrit)):
            sol = mgrit[j].u[0]
            diffs_vector = np.zeros(len(sol))
            for i in range(len(sol)):
                u_e = exact_sol(x=mgrit[j].problem[0].x, t=mgrit[j].problem[0].t[i])
                diffs_vector[i] = np.linalg.norm(sol[i].get_values() - u_e, np.inf)
            save_vecs.append(diffs_vector.copy())
            ax1.plot(heat0.t, diffs_vector, linetype[j], label=labels[j], color=colors[j], lw=lw, markevery=[],
                     markeredgewidth=3, markerfacecolor=mfc[j], marker=marker[j], markersize=ms, )
            count += 1
        diffs_vector = np.zeros(len(sol))
        for i in range(len(sol)):
            u_e = exact_sol(x=heat0.x, t=heat0.t[i])
            diffs_vector[i] += abs(ts[i].get_values() - u_e).max()
        ax1.plot(heat0.t, diffs_vector, label='time-stepping', color='blue', lw=lw, markevery=[],
                 markeredgewidth=3, markerfacecolor='blue', marker='x', markersize=ms)
        val = len(heat0.t) / 7
        for i in range(5):
            ax1.plot(heat0.t, save_vecs[i], color=[0, 0, 0, 0], lw=lw, markevery=[int((i + 1) * val)],
                     markeredgewidth=3, markerfacecolor=mfc[i], marker=marker[i], markeredgecolor=colors[i],
                     markersize=ms)
        ax1.plot(heat0.t, diffs_vector, color=[0, 0, 0, 0], lw=lw, markevery=[int(6 * val)],
                 markeredgewidth=3, markerfacecolor='blue', marker='x', markeredgecolor='blue', markersize=ms)
        ax1.set_xlabel('time', fontsize=fonts, weight='bold')
        ax1.set_ylabel('L-infinity norm of error', fontsize=fonts, weight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=fonts)  # , weight='bold')
        ax1.legend(loc='upper right', prop={'size': fonts, 'weight': 'bold'})
        plt.savefig("example_1_error.png", bbox_inches='tight')
        plt.show()

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
    runtime = np.zeros(5)
    iters = np.zeros(5)

    # V-cycle, FCF-relaxation
    for i in range(iterations):
        mgrit_V_FCF = Mgrit(problem=problem, cf_iter=1, cycle_type='V', random_init_guess=True,
                            nested_iteration=False)
        res = mgrit_V_FCF.solve()
        runtime[0] += res['time_setup'] + res['time_solve']
        iters[0] += len(res['conv'])

    # # V-cycle, FCFCF-relaxation, BDF1
    for i in range(iterations):
        mgrit_V_FCFCF = Mgrit(problem=problem, cf_iter=2, cycle_type='V', random_init_guess=True,
                              nested_iteration=False)
        res = mgrit_V_FCFCF.solve()
        runtime[1] += res['time_setup'] + res['time_solve']
        iters[1] += len(res['conv'])

    # F-cycle, F-relaxation
    for i in range(iterations):
        mgrit_F_F = Mgrit(problem=problem, cf_iter=0, cycle_type='F', random_init_guess=True,
                          nested_iteration=False)
        res = mgrit_F_F.solve()
        runtime[2] += res['time_setup'] + res['time_solve']
        iters[2] += len(res['conv'])

    # F-cycle, FCF-relaxation
    for i in range(iterations):
        mgrit_F_FCF = Mgrit(problem=problem, cf_iter=1, cycle_type='F', random_init_guess=True,
                            nested_iteration=False)
        res = mgrit_F_FCF.solve()
        runtime[3] += res['time_setup'] + res['time_solve']
        iters[3] += len(res['conv'])

    #  F-cycle, FCFCF-relaxation
    for i in range(iterations):
        mgrit_F_FCFCF = Mgrit(problem=problem, cf_iter=2, cycle_type='F', random_init_guess=True,
                              nested_iteration=False)
        res = mgrit_F_FCFCF.solve()
        runtime[4] += res['time_setup'] + res['time_solve']
        iters[4] += len(res['conv'])

    # Save and print results
    if MPI.COMM_WORLD.Get_rank() == 0:
        save_runtime = runtime / iterations
        save_iters = iters / iterations
        print('V-cycle with FCF-relaxation:', save_iters[0], save_runtime[0])
        print('V-cycle with FCFCF-relaxation:', save_iters[1], save_runtime[1])
        print('F-cycle with F-relaxation:', save_iters[2], save_runtime[2])
        print('F-cycle with FCF-relaxation:', save_iters[3], save_runtime[3])
        print('F-cycle with FCFCF-relaxation:', save_iters[4], save_runtime[4])
        print(save_iters)

    if MPI.COMM_WORLD.Get_size() == 1:
        ts = [heat0.vector_t_start.clone()]
        for i in range(1, len(heat0.t)):
            ts.append(heat0.step(u_start=ts[-1], t_start=heat0.t[i - 1], t_stop=heat0.t[i]))
        plot_error(mgrit=[mgrit_V_FCF, mgrit_V_FCFCF, mgrit_F_F, mgrit_F_FCF, mgrit_F_FCFCF], ts=ts)


if __name__ == '__main__':
    main()
