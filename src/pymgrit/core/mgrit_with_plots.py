import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

from pymgrit.core.mgrit import Mgrit


class MgritWithPlots(Mgrit):
    def __init__(self, *args, **kwargs) -> None:
        """
        Cumstomized MGRIT constructor
        :param args:
        :param kwargs:
        """
        # Call parent constructor
        super(MgritWithPlots, self).__init__(*args, **kwargs)

    def plot(self, plot_function):
        if self.comm_time.rank == 0:
            if self.spatial_parallel:
                if self.comm_space == 0:
                    plot_function()
            else:
                plot_function()

    def plot_convergence(self, save_name=None, fig_size_x=6.4, fig_size_y=4.8, dpi=100, text_size=15):
        def plot_conv():
            fig = plt.figure(figsize=(fig_size_x, fig_size_y), dpi=dpi, constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0)
            iters = np.arange(1, len(self.conv[np.where(self.conv != 0)]) + 1)
            plt.semilogy(iters, self.conv[np.where(self.conv != 0)], 'o-', lw=int(text_size / 2), markersize=text_size)
            plt.xticks(iters)
            plt.xlabel('iter #', size=text_size, weight='bold')
            plt.ylabel('residual norm', size=text_size, weight='bold')
            plt.yticks(size=text_size, weight='bold')
            plt.xticks(size=text_size, weight='bold')
            if save_name is not None:
                plt.savefig(save_name)
            plt.show()

        self.plot(plot_function=plot_conv)

    def plot_parallel_distribution(self, time_procs=None, save_name=None, fig_size_x=6.4, fig_size_y=4.8, dpi=100,
                                   text_size=15):

        if time_procs is None:
            time_procs = self.comm_time_size

        def plot_parallel_dist():
            fig = plt.figure(figsize=(fig_size_x, fig_size_y), dpi=dpi, constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0)
            ax = plt.gca()

            size_lvl = []
            left_lower = []
            boxes = []

            colors = list(mcolors.BASE_COLORS)
            if 'k' in colors: colors.remove('k')

            if time_procs > len(colors):
                for i in range(time_procs - len(colors)):
                    colors.append(list(np.random.rand(3)))

            nt = len(self.problem[0].t)

            split = self.split_into(number_points=nt, number_processes=time_procs)

            for i in range(len(split)):
                left_lower.append([np.sum(split[:i]), -0.2])

            for i in range(self.lvl_max):
                for j in range(len(split)):
                    if i != 0:
                        boxes.append(FancyBboxPatch(
                            [a + b for a, b in zip(left_lower[j], [-0.25, i])], split[j] - 0.25, 0.4,
                            boxstyle="round,pad=-0.0040,rounding_size=0.015", fc=colors[j],
                            label='Process ' + str(j)))
                    else:
                        boxes.append(FancyBboxPatch(
                            [a + b for a, b in zip(left_lower[j], [-0.25, i])], np.sum(split) - 0.25, 0.4,
                            boxstyle="round,pad=-0.0040,rounding_size=0.015", fc=colors[j],
                            label='Process ' + str(j)))
                        break
            for item in boxes:
                ax.add_patch(item)

                legend = plt.legend(handles=boxes[-time_procs:], fontsize=text_size, loc='center left',
                                    bbox_to_anchor=(1, 0.5), prop={'weight': 'bold', 'size': text_size})

            for i in range(self.lvl_max):
                if i != self.lvl_max -1:
                    tmp_coarse = np.where(np.in1d(self.problem[0].t, self.problem[i + 1].t))[0]
                tmp_fine = np.where(np.in1d(self.problem[0].t, self.problem[i].t))[0]
                plt.plot([0, nt - 1], [i, i], color='k')

                for j in range(nt):
                    if j in tmp_coarse:
                        plt.plot([j, j],[((self.lvl_max - 1) - i) + 0.1, ((self.lvl_max - 1) - i) - 0.1], color='k')
                    elif j in tmp_fine:
                        plt.plot([j, j],[((self.lvl_max - 1) - i) + 0.05, ((self.lvl_max - 1) - i) - 0.05],color='k')

            [s.set_visible(False) for s in ax.spines.values()]
            ax.set_xticks([])
            ax.set_yticks(np.arange(0, self.lvl_max, 1).tolist())
            ax.set_yticklabels(['Level ' + str(i) for i in range(self.lvl_max - 1, -1, -1)])
            plt.yticks(size=text_size, weight='bold')
            if save_name is not None:
                plt.savefig(save_name, dpi=dpi)
            plt.show()

        self.plot(plot_function=plot_parallel_dist)

    def plot_cycle(self, iterations=1, save_name=None, fig_size_x=6.4, fig_size_y=4.8, dpi=100, text_size=15):

        def plot_cyc():
            def mask(len, result):
                strings = np.array([''] * len, dtype=object)
                strings[np.where(result < 0)[0]] = 'F' + self.cf_iter * 'CF'
                strings[np.where(result == 0)[0]] = 'S'
                strings[np.where(result > 0)[0]] = 'F'
                return strings

            def plot_part(iter, len_res, len_nes, result, nest_it=False):
                if nest_it:
                    x_axis = np.arange(0, len_nes, 1)
                    strings = mask(len=len_nes, result=result)
                else:
                    x_axis = np.arange(iter * len_res + len_nes, len_nes + (iter + 1) * len_res, 1)
                    strings = mask(len=len_res, result=result)
                if iter < iterations - 1 or nest_it:
                    plt.axvline(x=x_axis[-1] + 0.4, color='k', linestyle='--')
                plt.plot(x_axis, np.abs(result), color='k')
                if iter != 0:
                    strings[0] = strings[0][1:]
                for i in range(len(result) if not nest_it else len(result) - 1):
                    plt.text(x_axis[i], np.abs(result[i]), strings[i], size=text_size, weight='bold', ha="center",
                             va="center",
                             bbox=dict(boxstyle="round", facecolor='white', edgecolor='k'))
                return np.average(x_axis)

            fig = plt.figure(figsize=(fig_size_x, fig_size_y), dpi=dpi, constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0)
            ax = plt.gca()

            result_nes = result_iter = np.zeros(0)
            x_ticks = []
            if self.nes_it:
                for i in range(self.lvl_max):
                    result_nes = np.hstack((result_nes, np.linspace(-i, i, 2 * i + 1)))
                x_ticks.append(plot_part(iter=0, len_res=0, len_nes=len(result_nes), result=result_nes, nest_it=True))
            if self.cycle_type == 'V':
                result_iter = np.hstack(
                    (result_iter, np.linspace(-self.lvl_max + 1, self.lvl_max - 1, 2 * (self.lvl_max - 1) + 1)))
            elif self.cycle_type == 'F':
                result_iter = np.hstack((result_iter, np.linspace(-self.lvl_max + 1, 0, self.lvl_max)))[:-1]
                for i in range(self.lvl_max):
                    result_iter = np.hstack((result_iter, np.linspace(-i, i, 2 * i + 1)))
            for iter in range(0, iterations):
                x_ticks.append(
                    plot_part(iter=iter, len_res=len(result_iter), len_nes=len(result_nes), result=result_iter,
                              nest_it=False))

            [s.set_visible(False) for s in ax.spines.values()]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                ['Nested iteration'] + ['Iteration ' + str(i) for i in range(iterations)] if self.nes_it else [
                    'Iteration ' + str(i) for i in range(iterations)])
            ax.set_yticks(np.arange(0, self.lvl_max, 1).tolist())
            ax.set_yticklabels(['Level ' + str(i) for i in range(self.lvl_max - 1, -1, -1)])
            # ax.tick_params(axis='both', which='both', length=0)
            plt.yticks(size=text_size, weight='bold')
            plt.xticks(size=text_size, weight='bold')
            plt.xlabel(' ', size=text_size, weight='bold')
            plt.ylabel(' ', size=text_size, weight='bold')
            if save_name is not None:
                plt.savefig(save_name)
            plt.show()

        self.plot(plot_function=plot_cyc)
