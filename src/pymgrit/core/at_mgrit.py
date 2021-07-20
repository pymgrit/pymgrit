"""
AT-MGRIT solver in FAS formulation
"""
import time
import logging
import sys
from operator import itemgetter
from itertools import groupby
import numpy as np

from mpi4py import MPI

from pymgrit.core.mgrit import Mgrit


class AtMgrit(Mgrit):
    def __init__(self, k, *args, **kwargs):
        """
        Constructor

        :param k: Distance of the local coarse grids
        :param args:
        :param kwargs:
        """
        self.k = k
        self.comm_black = None
        self.comm_green = None
        self.local_coarse_grid = None
        self.c_points_per_proc = None
        super().__init__(*args, **kwargs)

    def forward_solve(self, lvl: int) -> None:
        """
        Solves local coarse grid problems on the coarsest level

        :param lvl: AT-MGRIT level
        """

        runtime_fs = time.time()
        if self.lvl_max != 1:
            if self.comm_time_size != 1:
                if self.cpts[lvl].size > 0:
                    data_black = self.comm_black.allgather([
                        self.global_t[lvl][self.cpts[lvl]][-1],
                        self.g[lvl][-1].pack(),
                        self.u[lvl][-1].pack()
                    ])

                    data_green = self.comm_green.bcast(data_black, root=0)

                    data = data_black + data_green

                    tmp_uu = self.problem[lvl].vector_t_start.clone()
                    for i in range(len(data)):
                        if data[i][0] in self.local_coarse_grid:
                            index = np.where(data[i][0] == self.local_coarse_grid)[0][0]
                            if index == 0:
                                tmp_uu.unpack(data[i][2])
                            self.g_coarsest[index].unpack(data[i][1])

                    for i in range(1, len(self.local_coarse_grid)):
                        tmp_uu = self.g_coarsest[i] + self.step[lvl](u_start=tmp_uu,
                                                                     t_start=self.local_coarse_grid[i - 1],
                                                                     t_stop=self.local_coarse_grid[i])
                    if self.comm_time_rank != 0:
                        self.u[lvl][self.index_local_c[lvl][0]] = tmp_uu
                    else:
                        if self.c_points_per_proc[0] == 1:
                            self.u[lvl][self.index_local_c[lvl][0]] = tmp_uu
                        else:
                            self.u[lvl][self.index_local_c[lvl][1]] = tmp_uu

            else:
                tmp_u_arr = [item.clone() for item in self.u[lvl]]
                for point in range(len(self.global_t[lvl])):
                    tmp_u = tmp_u_arr[max(0, point - self.k + 1)]
                    for i in range(max(1, point - self.k + 2), point + 1):
                        tmp_u = self.g[lvl][i] + self.step[lvl](u_start=tmp_u,
                                                                t_start=self.global_t[lvl][i - 1],
                                                                t_stop=self.global_t[lvl][i])
                    self.u[lvl][point] = tmp_u

        logging.debug(f"Forward solve on {self.comm_time_rank} took {time.time() - runtime_fs} s")

    def setup_points_and_comm_info(self, lvl: int) -> None:
        """
        Computes local grid information for level *lvl*.
        Computes which process holds the previous and next point for each lvl and process.

        :param lvl: MGRIT level
        """
        self.global_t.append(np.copy(self.problem[lvl].t))
        points_time = np.size(self.global_t[lvl])
        all_pts = np.array(range(0, points_time))

        # Compute points per process
        # First level: Divide the points evenly
        # Other levels: Depends on the first level
        if lvl == 0:
            block_size_this_lvl, first_i_this_lvl = self.split_points(length=points_time,
                                                                      size=self.comm_time_size,
                                                                      rank=self.comm_time_rank)
            all_pts = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
            self.int_start = self.global_t[lvl][all_pts[0]]
            self.int_stop = self.global_t[lvl][all_pts[-1]]
        else:
            all_pts = np.where((self.global_t[lvl] >= self.int_start) & (self.global_t[lvl] <= self.int_stop))[0]

        # Compute C- and F-points
        if lvl != self.lvl_max - 1:
            all_cpts = np.where(np.in1d(self.problem[lvl].t, self.problem[lvl + 1].t))[0]
        else:
            all_cpts = np.array(range(0, points_time, self.m[lvl]))
        all_fpts = np.array(list(set(np.array(range(0, points_time))) - set(all_cpts)))
        cpts = np.sort(np.array(list(set(all_pts) - set(all_fpts)), dtype='int'))
        fpts = np.array(list(set(all_pts) - set(cpts)))
        fpts2 = np.array([item for sublist in np.array([np.array(xi, dtype=object) for xi in np.array(
            [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(fpts), lambda x: x[0] - x[1])], dtype=object)],
                                                       dtype=object)[::-1] for item in sublist])

        # Add ghost point if needed, set time interval
        with_ghost_point = False
        if self.comm_time_rank != 0 and all_pts.size > 0:
            with_ghost_point = True
            tmp = np.zeros(len(all_pts) + 1, dtype=int)
            tmp[0] = all_pts[0] - 1
            tmp[1:] = all_pts
            all_pts_with_ghost = tmp
        else:
            all_pts_with_ghost = all_pts

        self.t[lvl] = self.global_t[lvl][all_pts_with_ghost]
        self.cpts.append(cpts)

        # Communication in F-relax
        self.comm_front.append(bool(fpts.size > 0 and fpts[np.argmin(fpts)] - 1 in all_fpts))
        self.comm_back.append(bool(fpts.size > 0 and fpts[np.argmax(fpts)] + 1 in all_fpts))

        # Setup local indices
        self.index_local.append(np.nonzero(all_pts[:, None] == all_pts_with_ghost)[1])
        self.index_local_c.append(np.nonzero(cpts[:, None] == all_pts_with_ghost)[1])
        self.index_local_f.append(np.nonzero(fpts2[:, None] == all_pts_with_ghost)[1])

        # Communication before and after C- and F-relax
        self.first_is_c_point.append(
            bool(all_pts.size > 0 and all_pts[0] in cpts and all_pts[0] != 0 and all_pts[0] - 1 in all_fpts))
        self.first_is_f_point.append(bool(all_pts.size > 0 and all_pts[0] in fpts2 and all_pts[0] - 1 in all_cpts))
        self.last_is_c_point.append(bool(
            all_pts.size > 0 and all_pts[-1] in cpts and all_pts[-1] != points_time - 1 and all_pts[
                -1] + 1 in all_fpts))
        self.last_is_f_point.append(bool(
            all_pts.size > 0 and all_pts[-1] in fpts2 and all_pts[-1] != points_time - 1 and all_pts[
                -1] + 1 in all_cpts))

        # Setup communication info
        split = self.global_t[0][
            np.cumsum(self.split_into(number_points=len(self.global_t[0]), number_processes=self.comm_time_size)) - 1]
        tmp_send_to = -99
        tmp_get_from = -99
        if len(all_pts_with_ghost) > 0:
            if self.t[lvl][-1] != self.global_t[lvl][-1]:
                last_point_next = self.global_t[lvl][np.argwhere(self.global_t[lvl] == self.t[lvl][-1])[0][0] + 1]
                tmp_send_to = np.searchsorted(split, last_point_next)
            if with_ghost_point or self.t[lvl][0] != self.global_t[0][0]:
                tmp_get_from = np.searchsorted(split, self.t[lvl][0])
        self.send_to.append(tmp_send_to)
        self.get_from.append(tmp_get_from)

        # Coarsest level communication
        if lvl == self.lvl_max - 1:

            end_points = np.cumsum(
                self.split_into(number_points=len(self.global_t[0]), number_processes=self.comm_time_size)) - 1
            split = self.global_t[0][end_points]
            self.comm_coarsest_level = np.array([np.min(np.where((item <= split))) for item in self.global_t[-1]])

            values, self.c_points_per_proc = np.unique(self.comm_coarsest_level, return_counts=True)

            if np.any(self.c_points_per_proc[1:] > 1) or self.c_points_per_proc[0] > 2:
                if self.comm_time_size != 1:
                    print('Too many points on proc')
                    sys.exit()

            if self.c_points_per_proc[0] == 2:
                tmp_comm_coarsest_level = self.comm_coarsest_level[1:]
            else:
                tmp_comm_coarsest_level = self.comm_coarsest_level

            # New communicators for MGRIT-DD
            if self.comm_time_rank in tmp_comm_coarsest_level:
                idx = np.where(self.comm_time_rank == tmp_comm_coarsest_level)[0][0]
                color_black = idx // self.k
                if idx % self.k == self.k - 1:
                    color_green = color_black + 1
                else:
                    color_green = color_black
                if self.comm_time_rank != 0:
                    self.local_coarse_grid = np.array(
                        self.global_t[lvl][max(0, self.cpts[lvl][0] - self.k + 1):self.cpts[lvl][0] + 1])
                else:
                    if self.c_points_per_proc[0] == 1:
                        self.local_coarse_grid = np.array(
                            self.global_t[lvl][max(0, self.cpts[lvl][0] - self.k + 1):self.cpts[lvl][0] + 1])
                    else:
                        self.local_coarse_grid = np.array(
                            self.global_t[lvl][max(0, self.cpts[lvl][0] - self.k + 1):self.cpts[lvl][1] + 1])

            else:
                color_black = MPI.UNDEFINED
                color_green = MPI.UNDEFINED

            self.comm_black = self.comm_time.Split(color=color_black, key=self.comm_time_rank)
            self.comm_green = self.comm_time.Split(color=color_green, key=self.comm_time_rank)

    def create_coarsest_level(self) -> None:
        """
        Creates vectors u and g for coarsest level
        """
        if self.local_coarse_grid is not None:
            self.g_coarsest = [self.problem[-1].vector_template.clone_zero() for _ in
                               range(len(self.local_coarse_grid))]
        self.u_coarsest = [self.problem[-1].vector_template.clone_zero() for _ in range(len(self.global_t[-1]))]
        self.u_coarsest[0] = self.problem[-1].vector_t_start.clone()

    def ouput_run_information(self) -> None:
        """
        Outputs information of AT-MGRIT run.
        """
        msg = ['Run parameter overview',
               '  ' + '{0: <25}'.format(f'time interval') + ' : ' + '[' + str(self.problem[0].t[0]) + ', ' + str(
                   self.problem[0].t[-1]) + ']',
               '  ' + '{0: <25}'.format(f'number of time points ') + ' : ' + str(len(self.problem[0].t)),
               '  ' + '{0: <25}'.format(f'max dt ') + ' : ' + str(
                   np.max(self.problem[0].t[1:] - self.problem[0].t[:-1])),
               '  ' + '{0: <25}'.format(f'number of levels') + ' : ' + str(self.lvl_max),
               '  ' + '{0: <25}'.format(f'coarsening factors') + ' : ' + str(self.m[:-1]),
               '  ' + '{0: <25}'.format(f'relaxation weight') + ' : ' + str(self.weight_c),
               '  ' + '{0: <25}'.format(f'cf_iter') + ' : ' + str(self.cf_iter),
               '  ' + '{0: <25}'.format(f'nested iteration') + ' : ' + str(self.nes_it),
               '  ' + '{0: <25}'.format(f'cycle type') + ' : ' + str(self.cycle_type),
               '  ' + '{0: <25}'.format(f'stopping tolerance') + ' : ' + str(self.tol),
               '  ' + '{0: <25}'.format(f'time communicator size') + ' : ' + str(self.comm_time_size),
               '  ' + '{0: <25}'.format(f'space communicator size') + ' : ' + str(self.comm_space_size),
               '  ' + '{0: <25}'.format(f'distance') + ' : ' + str(self.k)]
        self.log_info(message='\n'.join(msg))
