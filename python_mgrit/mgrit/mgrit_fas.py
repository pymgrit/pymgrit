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

    def __init__(self, problem, grid_transfer, it=10, tol=1e-7, nested_iteration=True, cf_iter=1, cycle_type='V',
                 comm_time=MPI.COMM_WORLD, comm_space=None):
        """
        """
        logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S',
                            level=logging.INFO, stream=sys.stdout)

        self.problem = problem
        self.comm_time = comm_time
        self.comm_space = comm_space
        if self.comm_time.Get_rank() == 0:
            logging.info(f"Start setup")

        runtime_setup_start = time.time()

        # dummies
        self.lvl_max = len(problem)
        # self.u = []
        self.step = []
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

        for lvl in range(self.lvl_max):
            # self.u.append(problem[lvl].u)
            self.step.append(problem[lvl].step)
            self.v.append([object] * len(self.problem[lvl].u))
            self.g.append([object] * len(self.problem[lvl].u))
            for time_step in range(len(self.problem[lvl].u)):
                if lvl == 0:
                    self.v[lvl][time_step] = None
                else:
                    self.v[lvl][time_step] = self.problem[lvl].u[time_step].clone_zeros()
                self.g[lvl][time_step] = self.problem[lvl].u[time_step].clone_zeros()
            self.t.append(problem[lvl].t)

        if self.lvl_max == 1:
            self.m.append(1)

        for lvl in range(self.lvl_max - 1):
            self.m.append(int((np.size(self.t[lvl]) - 1) / (np.size(self.t[lvl + 1]) - 1)))
            self.proc_data.append(
                self.setup_points(nt=np.size(self.t[lvl]), m=self.m[lvl], size=self.comm_time.Get_size(),
                                  rank=self.comm_time.Get_rank()))
            self.restriction.append(grid_transfer[lvl].restriction)
            self.interpolation.append(grid_transfer[lvl].interpolation)

        if nested_iteration:
            self.nested_iteration()

        runtime_setup_stop = time.time()

        if self.comm_time.Get_rank() == 0:
            logging.info(f"Setup took {runtime_setup_stop - runtime_setup_start} s")

    def iteration(self, lvl, cycle_type, iteration, first_f):
        """
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            self.problem[lvl].u = self.comm_time.bcast(self.problem[lvl].u, root=0)
            return

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for i in range(self.cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            if i != self.cf_iter - 1:
                self.exchange(lvl=lvl)

        self.exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        self.exchange(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def f_relax(self, lvl):
        """
        """
        tmp_send = False
        req_s = None
        rank = self.comm_time.Get_rank()
        if len(self.proc_data[lvl]['fpts']) > 0:
            for i in np.nditer(self.proc_data[lvl]['fpts']):
                if i == np.min(self.proc_data[lvl]['fpts']) and self.proc_data[lvl]['comm_front']:
                    # req = self.comm_time.irecv(source=rank - 1, tag=self.proc_data[lvl]['all_pts'][0] - 1)
                    # self.problem[lvl].u[self.proc_data[lvl]['all_pts'][0] - 1] = req.wait()
                    self.problem[lvl].u[self.proc_data[lvl]['all_pts'][0] - 1] = self.comm_time.recv(source=rank - 1,
                                                                                                     tag=
                                                                                                     self.proc_data[
                                                                                                         lvl][
                                                                                                         'all_pts'][
                                                                                                         0] - 1)
                # print('g:', self.g[lvl][i].current, 'u:', self.step[lvl](index=i).current)
                self.problem[lvl].u[i] = self.g[lvl][i] + self.step[lvl](index=i)
                if i == np.max(self.proc_data[lvl]['fpts']) and self.proc_data[lvl]['comm_back']:
                    tmp_send = True
                    req_s = self.comm_time.isend(self.problem[lvl].u[self.proc_data[lvl]['all_pts'][-1]], dest=rank + 1,
                                                 tag=self.proc_data[lvl]['all_pts'][-1])
        if tmp_send:
            req_s.wait()

    def c_relax(self, lvl):
        """
        """
        if len(self.proc_data[lvl]['cpts']) > 0:
            for i in np.nditer(self.proc_data[lvl]['cpts']):
                if i != 0:
                    self.problem[lvl].u[i] = self.g[lvl][i] + self.step[lvl](index=i)

    def convergence_criteria(self, it):
        """
        """
        r = []
        r_norm = []
        cpts = np.array(range(0, len(self.t[0]), self.m[0]))
        block_size, first_i = self.split_points(length=np.size(self.t[0]) - 1, size=self.comm_time.Get_size(),
                                                rank=self.comm_time.Get_rank())
        for i in range(first_i + 1, (block_size + first_i + 1)):
            if i in cpts:
                r.append(self.step[0](index=i) - self.problem[0].u[i])
                r_norm.append(r[-1].norm())

        tmp = self.comm_time.allgather(r_norm)
        tmp = [item for sublist in tmp for item in sublist]

        for item in tmp:
            self.conv[it] += item ** 2

        self.conv[it] = self.conv[it] ** 0.5

    def forward_solve(self, lvl):
        """
        """
        if self.comm_time.Get_rank() == 0:
            for i in range(1, np.size(self.t[lvl])):
                self.problem[lvl].u[i] = self.g[lvl][i] + self.step[lvl](index=i)

    def fas_residual(self, lvl):
        """
        """
        for i in range(len(self.proc_data[lvl]['all_cpts'])):
            self.v[lvl + 1][i] = self.restriction[lvl](self.problem[lvl].u[self.proc_data[lvl]['all_cpts'][i]])
            self.problem[lvl + 1].u[i] = copy.deepcopy(self.v[lvl + 1][i])
        if np.size(self.proc_data[lvl]['cpts']) > 0:
            for i in range(0, np.size(self.proc_data[lvl]['cpts'])):
                j = i + self.proc_data[lvl]['first_i']
                if j != 0:
                    self.g[lvl + 1][i] = self.restriction[lvl](
                        self.g[lvl][self.proc_data[lvl]['cpts'][i]] \
                        - self.problem[lvl].u[self.proc_data[lvl]['cpts'][i]] \
                        + self.step[lvl](index=self.proc_data[lvl]['cpts'][i])) \
                        + self.v[lvl + 1][j] - self.step[lvl + 1](index=j)

        tmp = self.comm_time.allgather(self.g[lvl + 1][0:np.size(self.proc_data[lvl]['cpts'])])
        self.g[lvl + 1] = [item for sublist in tmp for item in sublist]

    def nested_iteration(self):
        self.forward_solve(self.lvl_max - 1)
        self.problem[-1].u = self.comm_time.bcast(self.problem[-1].u, root=0)

        for lvl in range(self.lvl_max - 2, -1, -1):
            tmp = [object] * len(self.problem[lvl + 1].u)
            for i in range(len(self.problem[lvl + 1].u)):
                tmp[i] = self.interpolation[lvl](u=self.problem[lvl + 1].u[i])
            self.problem[lvl].u[::self.m[lvl]] = tmp
            if lvl > 0:
                self.iteration(lvl, 'V', 0, True)

    def exchange(self, lvl):
        tmp = self.comm_time.allgather(
            self.problem[lvl].u[self.proc_data[lvl]['all_pts'][0]:(self.proc_data[lvl]['all_pts'][-1] + 1)] if len(
                self.proc_data[lvl]['all_pts']) > 0 else [])
        # tmp = [item for sublist in tmp for item in sublist]
        # for i in range(len(self.problem[lvl].u)):
        #    self.problem[lvl].u[i] = tmp[i]
        self.problem[lvl].u = [item for sublist in tmp for item in sublist]

    def f_exchange(self, lvl):
        rank = self.comm_time.Get_rank()
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][0] > 0 and
                np.size(np.where(self.proc_data[lvl]['cpts'] == self.proc_data[lvl]['all_pts'][0])[0]) == 1):
            self.problem[lvl].u[self.proc_data[lvl]['all_pts'][0] - 1] = self.comm_time.recv(source=rank - 1,
                                                                                             tag=self.proc_data[lvl][
                                                                                                     'all_pts'][
                                                                                                     0] - 1)
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][-1] < np.size(self.t[lvl]) - 1 and
                np.size(np.where(self.proc_data[lvl]['fpts'] == self.proc_data[lvl]['all_pts'][-1])[0]) == 1 and
                np.size(np.where(self.proc_data[lvl]['all_cpts'] == self.proc_data[lvl]['all_pts'][-1] + 1)[0]) == 1):
            self.comm_time.send(self.problem[lvl].u[self.proc_data[lvl]['all_pts'][-1]], dest=rank + 1,
                                tag=self.proc_data[lvl]['all_pts'][-1])

    def c_exchange(self, lvl):
        rank = self.comm_time.Get_rank()
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][0] > 0 and
                np.size(np.where(self.proc_data[lvl]['fpts'] == self.proc_data[lvl]['all_pts'][0])[0]) == 1 and
                np.size(np.where(self.proc_data[lvl]['all_cpts'] == self.proc_data[lvl]['all_pts'][0] - 1)[0]) == 1):
            self.problem[lvl].u[self.proc_data[lvl]['all_pts'][0] - 1] = self.comm_time.recv(source=rank - 1,
                                                                                             tag=self.proc_data[lvl][
                                                                                                     'all_pts'][
                                                                                                     0] - 1)

        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][-1] < np.size(self.t[lvl]) - 1 and
                np.size(np.where(self.proc_data[lvl]['cpts'] == self.proc_data[lvl]['all_pts'][-1])[0]) == 1):
            self.comm_time.send(self.problem[lvl].u[self.proc_data[lvl]['all_pts'][-1]], dest=rank + 1,
                                tag=self.proc_data[lvl]['all_pts'][-1])

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
        if self.comm_time.Get_rank() == 0:
            self.runtime_solve = runtime_solve_stop - runtime_solve_start
            logging.info(f"Solve took {self.runtime_solve} s")

        return {'u': self.problem[0].u, 'time': self.runtime_solve, 'conv': self.conv}

    def error_correction(self, lvl):
        e = []
        for i in range(self.proc_data[lvl]['first_i'],
                       self.proc_data[lvl]['first_i'] + self.proc_data[lvl]['block_size']):
            e.append(self.problem[lvl + 1].u[i] - self.v[lvl + 1][i])

        for i in range(len(e)):
            e[i] = self.interpolation[lvl](e[i])
            self.problem[lvl].u[self.proc_data[lvl]['cpts'][i]] = self.problem[lvl].u[self.proc_data[lvl]['cpts'][i]] + \
                                                                  e[i]

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

    def setup_points(self, nt, m, size, rank):
        all_pts = np.array(range(0, nt))
        block_size, first_i = self.split_points(length=np.size(all_pts), size=size, rank=rank)
        all_pts = all_pts[first_i:first_i + block_size]

        all_cpts = np.array(range(0, nt, m))
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

        all_cpts = np.array(range(0, nt, m))
        all_fpts = np.array(list(set(np.array(range(0, nt))) - set(all_cpts)))
        need_communication_front = False
        need_communication_back = False

        if np.size(fpts) > 0 and fpts[np.argmin(fpts)] - 1 in all_fpts:
            need_communication_front = True
        if np.size(fpts) > 0 and fpts[np.argmax(fpts)] + 1 in all_fpts:
            need_communication_back = True

        ret_dict = {
            'fpts': fpts2,
            'cpts': cpts,
            'all_cpts': all_cpts,
            'all_pts': all_pts,
            'first_i': first_i,
            'block_size': block_size,
            'comm_front': need_communication_front,
            'comm_back': need_communication_back
        }
        return ret_dict
