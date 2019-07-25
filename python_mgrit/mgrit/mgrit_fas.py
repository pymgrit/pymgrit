from mpi4py import MPI
import time
from operator import itemgetter
from itertools import *
import numpy as np
import logging
import copy
import sys


class MgritFas:
    """
    """

    def __init__(self, problem, grid_transfer, it=100, tol=1e-7, nested_iteration=True, cf_iter=1, cycle_type='V',
                 comm_time=MPI.COMM_WORLD, comm_space=None, debug_lvl=logging.INFO):
        """
        """
        logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S',
                            level=debug_lvl, stream=sys.stdout)

        self.problem = problem
        self.comm_time = comm_time
        self.comm_space = comm_space
        if self.comm_time.Get_rank() == 0:
            logging.info(f"Start setup")

        runtime_setup_start = time.time()

        self.lvl_max = len(problem)
        self.step = []
        self.u = []
        self.v = []
        self.g = []
        self.t = []
        self.proc_data = []
        self.m = []
        self.it = it
        self.tol = tol
        self.conv = np.zeros(it + 1)
        self.runtime_solve = 0
        self.cf_iter = cf_iter
        self.cycle_type = cycle_type
        self.restriction = []
        self.interpolation = []
        self.int_start = 0
        self.int_stop = 0
        self.g_coarsest = []
        self.u_coarsest = []
        self.comm_info = []

        for lvl in range(self.lvl_max):
            self.t.append(copy.deepcopy(problem[lvl].t))
            if lvl != self.lvl_max - 1:
                self.restriction.append(grid_transfer[lvl].restriction)
                self.interpolation.append(grid_transfer[lvl].interpolation)

        self.setup_comm_info()

        for lvl in range(self.lvl_max):
            if lvl < self.lvl_max - 1:
                self.m.append(int((np.size(self.t[lvl]) - 1) / (np.size(self.t[lvl + 1]) - 1)))
            else:
                self.m.append(1)
            self.proc_data.append(self.setup_points(lvl=lvl))
            self.step.append(problem[lvl].step)
            self.create_u(lvl=lvl)
            if lvl == 0:
                self.v.append(None)
            else:
                self.v.append(copy.deepcopy(self.u[lvl]))
            self.g.append(copy.deepcopy(self.u[lvl]))
            if lvl == self.lvl_max - 1:
                for i in range(len(self.problem[lvl].t)):
                    if i == 0:
                        self.u_coarsest.append(copy.deepcopy(self.problem[lvl].u))
                    else:
                        self.u_coarsest.append(self.problem[lvl].u.clone_zeros())
                    self.g_coarsest.append(self.problem[lvl].u.clone_zeros())

        print(nested_iteration)

        if nested_iteration:
            self.nested_iteration()

        runtime_setup_stop = time.time()

        if self.comm_time.Get_rank() == 0:
            logging.info(f"Setup took {runtime_setup_stop - runtime_setup_start} s")

    def create_u(self, lvl):

        self.u.append([object] * self.proc_data[lvl]['block_size_this_lvl'])
        for i in range(len(self.u[lvl])):
            self.u[lvl][i] = self.problem[lvl].u.clone_zeros()
        if self.comm_time.Get_rank() == 0:
            self.u[lvl][0] = copy.deepcopy(self.problem[lvl].u)

    def iteration(self, lvl, cycle_type, iteration, first_f):
        """
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for i in range(self.cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.f_exchange(lvl=lvl)
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def f_relax(self, lvl):
        """
        """
        runtime_f = time.time()
        tmp_send = False
        req_s = None
        rank = self.comm_time.Get_rank()
        if len(self.proc_data[lvl]['index_local_f']) > 0:
            for i in np.nditer(self.proc_data[lvl]['index_local_f']):
                if i == np.min(self.proc_data[lvl]['index_local_f']) and self.proc_data[lvl]['comm_front']:
                    self.u[lvl][0] = self.comm_time.recv(source=self.comm_info[lvl]['get_from'], tag=rank)

                self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                 t_start=self.t[lvl][i - 1],
                                                                 t_stop=self.t[lvl][i])

                if i == np.max(self.proc_data[lvl]['index_local_f']) and self.proc_data[lvl]['comm_back']:
                    tmp_send = True

                    req_s = self.comm_time.isend(self.u[lvl][-1], dest=self.comm_info[lvl]['send_to'],
                                                 tag=self.comm_info[lvl]['send_to'])
        if tmp_send:
            req_s.wait()

        logging.debug(f"F-relax on {rank} took {time.time() - runtime_f} s")

    def c_relax(self, lvl):
        """
        """
        runtime_c = time.time()
        if len(self.proc_data[lvl]['index_local_c']) > 0:
            for i in np.nditer(self.proc_data[lvl]['index_local_c']):
                if i != 0 or self.comm_time.Get_rank() != 0:
                    self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                     t_start=self.t[lvl][i - 1],
                                                                     t_stop=self.t[lvl][i])
        logging.debug(f"C-relax on {self.comm_time.Get_rank()} took {time.time() - runtime_c} s")

    def convergence_criteria(self, it):
        """
        """
        runtime_conv = time.time()
        r_norm = []

        self.f_exchange(lvl=0)
        self.c_exchange(lvl=0)

        if len(self.proc_data[0]['index_local_c']) > 0:
            for i in np.nditer(self.proc_data[0]['index_local_c']):
                if self.comm_time.Get_rank() != 0 or i != 0:
                    r = self.step[0](u_start=self.u[0][i - 1], t_start=self.t[0][i - 1], t_stop=self.t[0][i]) - self.u[0][i]
                    r_norm.append(r.norm())

        tmp = self.comm_time.allgather(r_norm)
        tmp = [item for sublist in tmp for item in sublist]

        for item in tmp:
            self.conv[it] += item ** 2

        self.conv[it] = self.conv[it] ** 0.5
        logging.debug(f"Convergence criteria on {self.comm_time.Get_rank()} took {time.time() - runtime_conv} s")

    def forward_solve(self, lvl):
        """
        """
        runtime_fs = time.time()
        if self.comm_time.Get_rank() == 0:
            for i in range(1, len(self.problem[lvl].t)):
                self.u_coarsest[i] = self.g_coarsest[i] + self.step[lvl](u_start=self.u_coarsest[i - 1],
                                                                         t_start=self.problem[lvl].t[i - 1],
                                                                         t_stop=self.problem[lvl].t[i])
        self.u_coarsest = self.comm_time.bcast(self.u_coarsest, root=0)
        if len(self.proc_data[lvl]['cpts']) > 0:
            self.u[lvl] = [self.u_coarsest[i] for i in self.proc_data[lvl]['cpts']]
            if self.comm_time.Get_rank() != 0:
                self.u[lvl] = [self.u[lvl][0]] + self.u[lvl]

        logging.debug(f"Forward solve on {self.comm_time.Get_rank()} took {time.time() - runtime_fs} s")

    def get_c_point(self, lvl):
        rank = self.comm_time.Get_rank()
        tmp_send = False
        tmp = None
        req_s = None

        if self.comm_info[lvl + 1]['send_to'] >= 0:
            # if rank != self.comm_time.Get_size() - 1 and len(self.proc_data[lvl]['index_local_c']) > 0:
            #print('send in get_c', self.comm_info[lvl + 1]['send_to'], rank)
            req_s = self.comm_time.isend(self.u[lvl][self.proc_data[lvl]['index_local_c'][-1]],
                                         dest=self.comm_info[lvl + 1]['send_to'],
                                         tag=rank)
            tmp_send = True

        # if rank != 0 and len(self.v[lvl + 1]) > 0:
        if self.comm_info[lvl + 1]['get_from'] >= 0:
            #print('recv in get_c', self.comm_info[lvl + 1]['get_from'], self.comm_info[lvl + 1]['get_from'])
            tmp = self.comm_time.recv(source=self.comm_info[lvl + 1]['get_from'],
                                      tag=self.comm_info[lvl + 1]['get_from'])

        if tmp_send:
            req_s.wait()
        return tmp

    def fas_residual(self, lvl):
        """
        """
        runtime_fas_res = time.time()
        tmp = self.get_c_point(lvl=lvl)
        rank = self.comm_time.Get_rank()

        if self.comm_time.Get_rank() != 0 and len(self.v[lvl + 1]) > 0:
            #print('tmp on rank', tmp, rank)
            self.v[lvl + 1][0] = self.restriction[lvl](tmp)

        for i in range(len(self.proc_data[lvl]['index_local_c'])):
            self.v[lvl + 1][i if rank == 0 else i + 1] = self.restriction[lvl](
                self.u[lvl][self.proc_data[lvl]['index_local_c'][i]])

        self.u[lvl + 1] = copy.deepcopy(self.v[lvl + 1])
        if np.size(self.proc_data[lvl]['index_local_c']) > 0:
            for i in range(len(self.proc_data[lvl]['index_local_c'])):
                if i != 0 or self.comm_time.Get_rank() != 0:
                    self.g[lvl + 1][self.proc_data[lvl + 1]['index_local'][i]] = self.restriction[lvl](
                        self.g[lvl][self.proc_data[lvl]['index_local_c'][i]]
                        - self.u[lvl][self.proc_data[lvl]['index_local_c'][i]]
                        + self.step[lvl](u_start=self.u[lvl][self.proc_data[lvl]['index_local_c'][i] - 1],
                                         t_start=self.t[lvl][self.proc_data[lvl]['index_local_c'][i] - 1],
                                         t_stop=self.t[lvl][self.proc_data[lvl]['index_local_c'][i]])) \
                                                                                 + self.v[lvl + 1][
                                                                                     self.proc_data[lvl + 1][
                                                                                         'index_local'][i]] \
                                                                                 - self.step[lvl + 1](
                        u_start=self.v[lvl + 1][self.proc_data[lvl + 1]['index_local'][i] - 1],
                        t_start=self.t[lvl + 1][self.proc_data[lvl + 1]['index_local'][i] - 1],
                        t_stop=self.t[lvl + 1][self.proc_data[lvl + 1]['index_local'][i]])

        if lvl == self.lvl_max - 2:
            tmp_g = self.comm_time.gather([self.g[lvl + 1][i] for i in self.proc_data[lvl + 1]['index_local_c']],
                                          root=0)
            tmp_u = self.comm_time.gather([self.u[lvl + 1][i] for i in self.proc_data[lvl + 1]['index_local_c']],
                                          root=0)
            if self.comm_time.Get_rank() == 0:
                self.g_coarsest = [item for sublist in tmp_g for item in sublist]
                self.u_coarsest = [item for sublist in tmp_u for item in sublist]

        logging.debug(f"Fas residual on {self.comm_time.Get_rank()} took {time.time() - runtime_fas_res} s")

    def nested_iteration(self):
        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.proc_data[lvl + 1]['index_local'])):
                self.u[lvl][self.proc_data[lvl]['index_local_c'][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.proc_data[lvl + 1]['index_local'][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl, 'V', 0, True)

    def f_exchange(self, lvl):
        runtime_ex = time.time()
        rank = self.comm_time.Get_rank()
        if self.proc_data[lvl]['first_is_c_point']:
            self.u[lvl][0] = self.comm_time.recv(source=self.comm_info[lvl]['get_from'], tag=rank)
        if self.proc_data[lvl]['last_is_f_point']:
            self.comm_time.send(self.u[lvl][-1], dest=self.comm_info[lvl]['send_to'],
                                tag=self.comm_info[lvl]['send_to'])
        logging.debug(f"Exchange on {self.comm_time.Get_rank()} took {time.time() - runtime_ex} s")

    def c_exchange(self, lvl):
        runtime_ex = time.time()
        rank = self.comm_time.Get_rank()
        if self.proc_data[lvl]['first_is_f_point']:
            #print('c_relax recv from', self.comm_info[lvl]['get_from'], rank)
            self.u[lvl][0] = self.comm_time.recv(source=self.comm_info[lvl]['get_from'], tag=rank)
        if self.proc_data[lvl]['last_is_c_point']:
            #print('c_relax send from', self.comm_info[lvl]['send_to'], rank)
            self.comm_time.send(self.u[lvl][-1], dest=self.comm_info[lvl]['send_to'],
                                tag=self.comm_info[lvl]['send_to'])
        logging.debug(f"Exchange on {self.comm_time.Get_rank()} took {time.time() - runtime_ex} s")

    def solve(self):
        """
        """
        if self.comm_time.Get_rank() == 0:
            logging.info("Start solve")

        runtime_solve_start = time.time()
        for iteration in range(self.it):

            time_it_start = time.time()
            self.iteration(lvl=0, cycle_type=self.cycle_type, iteration=iteration, first_f=True)
            time_it_stop = time.time()
            self.convergence_criteria(it=iteration + 1)

            if self.comm_time.Get_rank() == 0:
                if iteration == 0:
                    logging.info('{0: <7}'.format(f"step {iteration + 1}") + '{0: <30}'.format(
                        f" | con: {self.conv[iteration + 1]}") + '{0: <35}'.format(
                        f" | con-fac: -") + '{0: <35}'.format(
                        f" | runtime: {time_it_stop - time_it_start} s"))
                else:
                    logging.info('{0: <7}'.format(f"step {iteration + 1}") + '{0: <30}'.format(
                        f" | con: {self.conv[iteration + 1]}") + '{0: <35}'.format(
                        f" | con-fac: {self.conv[iteration + 1] / self.conv[iteration]}") + '{0: <35}'.format(
                        f" | runtime: {time_it_stop - time_it_start} s"))

            if self.conv[iteration + 1] < self.tol:
                break

        runtime_solve_stop = time.time()

        solution = self.comm_time.gather([self.u[0][i] for i in self.proc_data[0]['index_local']], root=0)

        if self.comm_time.Get_rank() == 0:
            solution = [item for sublist in solution for item in sublist]
            self.runtime_solve = runtime_solve_stop - runtime_solve_start
            logging.info(f"Solve took {self.runtime_solve} s")

        return {'u': solution, 'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t}

    def error_correction(self, lvl):
        for i in range(len(self.proc_data[lvl]['index_local_c'])):
            e = self.interpolation[lvl](self.u[lvl + 1][self.proc_data[lvl]['index_local'][i]] - self.v[lvl + 1][
                self.proc_data[lvl]['index_local'][i]])
            self.u[lvl][self.proc_data[lvl]['index_local_c'][i]] = self.u[lvl][
                                                                       self.proc_data[lvl]['index_local_c'][i]] + e
        self.c_exchange(lvl=lvl)

    def split_points(self, length, size, rank):
        block_size = self.split_into(n=length, p=size)[rank]

        first_i = 0
        if block_size > 0:
            for i in range(size):
                if i == rank:
                    break
                first_i += self.split_into(length, size)[i]
        return block_size, first_i

    @staticmethod
    def split_into(n, p):
        return np.array([int(n / p + 1)] * (n % p) + [int(n / p)] * (p - n % p))

    def setup_points(self, lvl):
        nt = np.size(self.t[lvl])
        all_pts = np.array(range(0, nt))

        # Compute all pts per process
        # First level by splitting of points
        # other levels by time depending on first grid
        if lvl == 0:
            block_size_this_lvl, first_i_this_lvl = self.split_points(length=np.size(all_pts),
                                                                      size=self.comm_time.Get_size(),
                                                                      rank=self.comm_time.Get_rank())
            all_pts = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
            self.int_start = self.t[lvl][all_pts[0]]
            self.int_stop = self.t[lvl][all_pts[-1]]
        else:
            all_pts = np.where((self.t[lvl] >= self.int_start) & (self.t[lvl] <= self.int_stop))[0]

        all_cpts = np.array(range(0, nt, self.m[lvl]))

        all_fpts = np.array(list(set(np.array(range(0, nt))) - set(all_cpts)))

        cpts = np.sort(np.array(list(set(all_pts) - set(all_fpts)), dtype='int'))
        block_size = np.size(cpts)
        if block_size > 0:
            first_i = np.where(cpts[0] == all_cpts)[0][0]
        else:
            first_i = 0
        fpts = np.array(list(set(all_pts) - set(cpts)))

        fpts2 = np.array([item for sublist in np.array([np.array(xi) for xi in np.asarray(
            [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(fpts), lambda x: x[0] - x[1])])])[::-1] for item
                          in
                          sublist])

        all_cpts = np.array(range(0, nt, self.m[lvl]))
        all_fpts = np.array(list(set(np.array(range(0, nt))) - set(all_cpts)))
        need_communication_front = False
        need_communication_back = False

        if np.size(fpts) > 0 and fpts[np.argmin(fpts)] - 1 in all_fpts:
            need_communication_front = True
        if np.size(fpts) > 0 and fpts[np.argmax(fpts)] + 1 in all_fpts:
            need_communication_back = True

        if len(all_pts) > 0 and all_pts[0] in cpts and all_pts[0] != 0:
            first_is_c_point = True
        else:
            first_is_c_point = False

        if len(all_pts) > 0 and all_pts[0] in fpts2 and all_pts[0] - 1 in all_cpts:
            first_is_f_point = True
        else:
            first_is_f_point = False

        if len(all_pts) > 0 and all_pts[-1] in cpts and all_pts[-1] != nt - 1:
            last_is_c_point = True
        else:
            last_is_c_point = False

        if len(all_pts) > 0 and all_pts[-1] in fpts2 and all_pts[-1] != nt - 1 and all_pts[-1] + 1 in all_cpts:
            last_is_f_point = True
        else:
            last_is_f_point = False

        if self.comm_time.Get_rank() != 0 and len(all_pts) > 0:
            tmp = np.zeros(len(all_pts) + 1, dtype=int)
            tmp[0] = all_pts[0] - 1
            tmp[1:] = all_pts
            all_pts_with_ghost = tmp
            self.t[lvl] = self.t[lvl][all_pts_with_ghost]
        else:
            all_pts_with_ghost = all_pts

        index_local_c = np.zeros_like(cpts)
        index_local_f = np.zeros_like(fpts2)
        index_local = np.zeros_like(all_pts)

        for i in range(len(cpts)):
            index_local_c[i] = np.where(cpts[i] == all_pts_with_ghost)[0]

        for i in range(len(fpts2)):
            index_local_f[i] = np.where(fpts2[i] == all_pts_with_ghost)[0]

        for i in range(len(all_pts)):
            index_local[i] = np.where(all_pts[i] == all_pts_with_ghost)[0]

        ret_dict = {
            'fpts': fpts2,
            'cpts': cpts,
            'all_cpts': all_cpts,
            'all_pts': all_pts,
            'first_i': first_i,
            'block_size': block_size,
            'comm_front': need_communication_front,
            'comm_back': need_communication_back,
            'block_size_this_lvl': len(all_pts_with_ghost),
            'index_local_c': index_local_c,
            'index_local_f': index_local_f,
            'index_local': index_local,
            'first_is_f_point': first_is_f_point,
            'first_is_c_point': first_is_c_point,
            'last_is_f_point': last_is_f_point,
            'last_is_c_point': last_is_c_point
        }
        return ret_dict

    def setup_comm_info(self):

        communication_info = []
        start = np.zeros(self.comm_time.Get_size())
        stop = np.zeros(self.comm_time.Get_size())
        for lvl in range(self.lvl_max):
            nt = np.size(self.t[lvl])
            all_pts = np.array(range(0, nt))
            this_level = np.zeros(0, dtype=int)
            for proc in range(self.comm_time.Get_size()):
                if lvl == 0:
                    block_size_this_lvl, first_i_this_lvl = self.split_points(length=np.size(all_pts),
                                                                              size=self.comm_time.Get_size(),
                                                                              rank=proc)
                    tmp = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
                    start[proc] = self.t[lvl][tmp[0]]
                    stop[proc] = self.t[lvl][tmp[-1]]
                else:
                    tmp = np.where((self.t[lvl] >= start[proc]) & (self.t[lvl] <= stop[proc]))[0]
                this_level = np.hstack((this_level, np.ones(len(tmp)) * proc)).astype(int)
            points = np.where(self.comm_time.Get_rank() == this_level)[0]
            if len(points) > 0:
                if points[0] == 0:
                    front = -99
                else:
                    front = this_level[points[0] - 1]
                if points[-1] == nt - 1:
                    back = -99
                else:
                    back = this_level[points[-1] + 1]
            else:
                front = -99
                back = -99
            #print('this level', self.comm_time.Get_rank(), lvl, this_level, front, back)
            self.comm_info.append({'send_to': back, 'get_from': front})
