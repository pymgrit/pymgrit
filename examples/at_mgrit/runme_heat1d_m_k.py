import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from mpi4py import MPI

from pymgrit.core.at_mgrit import AtMgrit
from pymgrit.heat.heat_1d import Heat1D

comm_world = MPI.COMM_WORLD

fonts = 32
lw = 4
ms = 12


def run_at_mgrit_two_level(m, save_file, k):
    heat_0 = Heat1D(x_start=0, x_end=3, nx=1025, a=1, init_cond=lambda x: np.sin(np.pi * x),
                    rhs=lambda x, t: - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t)),
                    t_start=0, t_stop=np.pi, nt=2 ** 14)

    heat_1 = Heat1D(x_start=0, x_end=3, nx=1025, a=1, init_cond=lambda x: np.sin(np.pi * x),
                    rhs=lambda x, t: - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t)),
                    t_interval=heat_0.t[::m])

    save_runs = []
    fac = 2
    rank = comm_world.Get_rank()
    while k <= len(heat_1.t):
        solver = AtMgrit(k=k, problem=[heat_0, heat_1], random_init_guess=True, nested_iteration=False,
                         logging_lvl=20, tol=1e-07, cf_iter=0)
        info = solver.solve()
        save_runs.append(info)
        k += fac
    if rank == 0:
        np.save(save_file, np.array(save_runs))


def plot_convergence(file_data, save_file):
    colors = ['green', 'black', 'orange', 'magenta', 'blue']
    marker = ['s', 'D', 'o', 's', 'D']
    mfc = ['green', 'black', 'white', 'white', 'white']
    data = np.load(file_data, allow_pickle=True)
    vals = [8, 12, 16, 128]
    distances = np.arange(2, 128 + 1, 2)
    ind = np.where(np.in1d(distances, vals))[0]
    conv = []
    for item in ind:
        conv.append(data[item]['conv'])

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)
    ax.set_yscale('log', basey=10)

    for i in range(len(conv)):
        plt.plot(np.arange(1, len(conv[i]) + 1, 1), conv[i], marker=marker[i], label='2-level, F, k=' + str(vals[i]),
                 lw=lw, color=colors[i], markersize=ms,
                 markeredgewidth=3, markerfacecolor=mfc[i])

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('iterations', fontsize=fonts, weight='bold')
    plt.ylabel('residual norm', fontsize=fonts, weight='bold')
    plt.xticks(size=fonts, weight='bold')
    plt.yticks(size=fonts, weight='bold')
    plt.legend(loc='lower left', prop={'size': fonts, 'weight': 'bold'})
    for axis in [ax.xaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


def load_and_plot_1(m, distance, path, marker, color):
    data = np.load(path, allow_pickle=True)
    distances = np.arange(distance, int(2 ** 14 / m) + 1, distance)
    conv = []
    for item in data:
        conv.append(len(item['conv']))
    conv = np.array(conv)
    plt.plot(distances[1:], conv[1:], marker=marker, label='2-level, F, m=' + str(m), lw=lw, color=color,
             markersize=ms,
             markeredgewidth=3)


def load_and_plot_2(m, distance, path, marker, color, fac=1.0):
    data = np.load(path, allow_pickle=True)
    distances = np.arange(distance, int(2 ** 14 / m) + 1, distance)
    x = distances / int(2 ** 14 / m)
    q = np.where(x <= fac)[0]
    conv = []
    for item in data:
        conv.append(len(item['conv']))
    conv = np.array(conv)
    plt.plot(x[1:len(q)], conv[1:len(q)], marker=marker, label='2-level, F, m=' + str(m), lw=lw, color=color,
             markersize=ms,
             markeredgewidth=3)


def plot_iterations_1(file_data, save_file):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)
    load_and_plot_1(m=128, distance=2, path=file_data, marker='d', color='blue')
    plt.xlabel('k', fontsize=fonts, weight='bold')
    plt.ylabel('iterations', fontsize=fonts, weight='bold')
    plt.xticks(size=fonts, weight='bold')
    plt.yticks(size=fonts, weight='bold')
    plt.legend(loc='upper right', prop={'size': fonts, 'weight': 'bold'})
    for axis in [ax.xaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


def plot_iterations_2(file_data_64, file_data_128, file_data_256, save_file):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True)

    load_and_plot_2(m=64, distance=2, path=file_data_64, marker='o', color='orange', fac=0.2)
    load_and_plot_2(m=128, distance=2, path=file_data_128, marker='d', color='blue', fac=0.2)
    load_and_plot_2(m=256, distance=2, path=file_data_256, marker='s', color='red', fac=0.2)
    plt.xlabel('k/(#C-points)', fontsize=fonts, weight='bold')
    plt.ylabel('iterations', fontsize=fonts, weight='bold')
    plt.xticks(size=fonts, weight='bold')
    plt.xticks([0.05, 0.10, 0.15, 0.2], size=fonts, weight='bold')
    plt.yticks(size=fonts, weight='bold')
    plt.legend(loc='upper right', prop={'size': fonts, 'weight': 'bold'})
    for axis in [ax.xaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    run_at_mgrit_two_level(m=64, save_file="results/at_mgrit_two_level_heat_64", k=2)
    run_at_mgrit_two_level(m=128, save_file="results/at_mgrit_two_level_heat_128", k=2)
    run_at_mgrit_two_level(m=256, save_file="results/at_mgrit_two_level_heat_256", k=2)

    # plot_convergence(file_data="results/at_mgrit_two_level_heat_128.npy", save_file="results/heat_conv.png")

    # # plot_iterations_1(file_data="results/at_mgrit_two_level_heat_128.npy", save_file="results/heat_diff_k.png")
    #
    # plot_iterations_2(file_data_64="results/at_mgrit_two_level_heat_64.npy",
    #                   file_data_128="results/at_mgrit_two_level_heat_128.npy",
    #                   file_data_256="results/at_mgrit_two_level_heat_256.npy",
    #                   save_file="results/heat_Pk.png")
