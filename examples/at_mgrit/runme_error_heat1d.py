import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from pymgrit.heat.heat_1d import Heat1D
from pymgrit.core.at_mgrit import AtMgrit


def compute_analytical_solution():
    x = np.linspace(0, 3, 1025)
    analytical_sol = np.zeros((len(heat0.t), 1025))
    for i in range(len(heat0.t)):
        analytical_sol[i] = np.sin(np.pi * x) * np.cos(heat0.t[i])
    return analytical_sol


def compute_ts_solution():
    u = [[] for _ in range(len(heat0.t))]
    u[0] = heat0.vector_t_start
    for i in range(1, len(heat0.t)):
        u[i] = heat0.step(u_start=u[i - 1], t_start=heat0.t[i - 1], t_stop=heat0.t[i])
    return np.array(u)


def compute_at_mgrit_solutions():
    at_mgrit_solutions = []
    for item in [8, 12, 16, 128]:
        solver = AtMgrit(k=item, problem=[heat0, heat1], random_init_guess=True, nested_iteration=False,
                         tol=1e-07, cf_iter=0)
        info = solver.solve()
        at_mgrit_solutions.append(np.array([item.get_values() for item in solver.u[0]]))
    return at_mgrit_solutions


def plot_error(sol_analytical, sol_ts, sol_at_mgrit, save_file):
    fonts = 28
    lw = 4
    ms = 12
    labels = [
        '2-level, F, k=8',
        '2-level, F, k=12',
        '2-level, F, k=16',
        '2-level, F, k=128',
        'time-stepping'
    ]
    colors = [
        'green',
        'black',
        'orange',
        'magenta',
        'blue'
    ]
    marker = ['s', 'D', 'o', 's', 'D']
    mfc = ['green', 'black', 'white', 'white', 'white']

    error_at_mgrit = []
    for j in range(len(sol_at_mgrit)):
        w = np.zeros(len(sol_ts))
        for i in range(len(sol_at_mgrit[j])):
            w[i] = np.linalg.norm(sol_analytical[i][1:-1] - sol_at_mgrit[j][i], np.inf)
        error_at_mgrit.append(w.copy())

    error_ts = np.zeros(len(sol_ts))
    for i in range(len(sol_ts)):
        error_ts[i] = np.linalg.norm(sol_analytical[i][1:-1] - sol_ts[i].get_values(), np.inf)

    dis = int(len(heat0.t) / (len(sol_at_mgrit) + 3))
    markers_on = [[(i + 1) * dis] for i in range(len(sol_at_mgrit) + 1)]

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 1, 1)

    count = 0
    for item in error_at_mgrit:
        ax1.plot(heat0.t, item, color=colors[count], lw=lw)
        count += 1
    ax1.plot(heat0.t, error_ts, color=colors[count], lw=lw)

    count = 0
    for item in error_at_mgrit:
        ax1.plot(heat0.t[markers_on[count]], item[markers_on[count]], color=colors[count], marker=marker[count],
                 label=labels[count], ms=ms, markerfacecolor=mfc[count], markeredgewidth=3, lw=lw)
        count += 1
    ax1.plot(heat0.t[markers_on[count]], error_ts[markers_on[count]], color=colors[count], marker=marker[count],
             label=labels[count], ms=ms, markerfacecolor=mfc[count], markeredgewidth=3, lw=lw)

    ax1.set_xlabel('time', fontsize=fonts, weight='bold')
    ax1.set_ylabel('L-infinity norm of error', fontsize=fonts, weight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=fonts)
    ax1.legend(loc='lower right', prop={'size': fonts - 3, 'weight': 'bold'})
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    heat0 = Heat1D(x_start=0, x_end=3, nx=1025, a=1, init_cond=lambda x: np.sin(np.pi * x),
                   rhs=lambda x, t: - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t)),
                   t_start=0, t_stop=np.pi, nt=2 ** 14)

    heat1 = Heat1D(x_start=0, x_end=3, nx=1025, a=1, init_cond=lambda x: np.sin(np.pi * x),
                   rhs=lambda x, t: - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t)),
                   t_interval=heat0.t[::128])

    comm_world = MPI.COMM_WORLD

    sol_ts = compute_ts_solution()
    sol_analytical = compute_analytical_solution()
    sols_at_mgrit = compute_at_mgrit_solutions()

    plot_error(sol_analytical=sol_analytical, sol_ts=sol_ts, sol_at_mgrit=sols_at_mgrit, save_file="results/heat_error")
