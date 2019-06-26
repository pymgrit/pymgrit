from mpi4py import MPI
from scipy import linalg as la
import time
from operator import itemgetter
from itertools import *
import numpy as np
import logging
import pathlib


class MgritFas:
    """
    """

    def __init__(self, problem, comm_time=MPI.COMM_WORLD, comm_space=None):
        """
        """
        logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S',
                            level=logging.INFO)

        self.problem = problem
        self.comm_time = comm_time
        self.comm_space = comm_space
        self.phi = self.problem.phi
        self.restriction = self.problem.restriction
        self.interpolation = self.problem.interpolation

        # dummies
        self.u = []
        self.v = []
        self.g = []
        self.t = []
        self.proc_data = []
        self.spatial_coarsening = []
        self.m = []
        self.it = 0
        self.tol = 0
        self.conv = []
        self.lvl_max = 0
        self.app = []
        self.setup_done = False
        self.runtime_solve = 0

    def setup_and_solve(self, lvl_max, m, it_max, tol, cf_iter=1, cycle_type='V', nested_iteration=True,
                        spatial_coarsening=False, spatial_max_lvl=0):
        """
        """
        self.setup(lvl_max=lvl_max, m=m, it=it_max, tol=tol, nested_iteration=nested_iteration,
                   spatial_coarsening=spatial_coarsening, spatial_max_lvl=spatial_max_lvl)
        self.solve(cf_iter=cf_iter, cycle_type=cycle_type)
        return self.u[0]

    def iteration(self, lvl, cf_iter, cycle_type, iteration, first_f):
        """
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            self.u[lvl] = self.comm_time.bcast(self.u[lvl], root=0)
            return

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for i in range(cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            if i != cf_iter - 1:
                self.exchange(lvl=lvl)

        self.exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cf_iter=cf_iter, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        self.exchange(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.iteration(lvl=lvl, cf_iter=cf_iter, cycle_type='V', iteration=iteration, first_f=False)

    def f_relax(self, lvl):
        """
        """
        rank = self.comm_time.Get_rank()
        if len(self.proc_data[lvl]['fpts']) > 0:
            for i in np.nditer(self.proc_data[lvl]['fpts']):
                if i == np.min(self.proc_data[lvl]['fpts']) and self.proc_data[lvl]['comm_front']:
                    req = self.comm_time.Irecv(self.u[lvl][self.proc_data[lvl]['all_pts'][0] - 1], rank - 1,
                                               tag=self.proc_data[lvl]['all_pts'][0] - 1)
                    req.Wait()
                self.u[lvl][i] = self.g[lvl][i] + self.phi(self.u[lvl][i - 1], self.t[lvl][i - 1], self.t[lvl][i],
                                                           self.app[lvl])
                if i == np.max(self.proc_data[lvl]['fpts']) and self.proc_data[lvl]['comm_back']:
                    self.comm_time.Isend(self.u[lvl][self.proc_data[lvl]['all_pts'][-1]], rank + 1,
                                         tag=self.proc_data[lvl]['all_pts'][-1])

    def c_relax(self, lvl):
        """
        """
        if len(self.proc_data[lvl]['cpts']) > 0:
            for i in np.nditer(self.proc_data[lvl]['cpts']):
                if i != 0:
                    self.u[lvl][i] = self.g[lvl][i] + self.phi(self.u[lvl][i - 1], self.t[lvl][i - 1], self.t[lvl][i],
                                                               self.app[lvl])

    def convergence_criteria(self, it):
        """
        """
        cpts = np.array(range(0, len(self.t[0]), self.m[0]))
        block_size, first_i = self.split_points(length=np.size(self.t[0]) - 1, size=self.comm_time.Get_size(),
                                                rank=self.comm_time.Get_rank())
        r = np.zeros((block_size, np.size(self.u[0], 1)))
        for i in range(first_i + 1, (block_size + first_i + 1)):
            if i in cpts:
                j = i - first_i - 1
                r[j] = self.phi(self.u[0][i - 1], self.t[0][i - 1], self.t[0][i], self.app[0]) - self.u[0][i]

        r_full = self.comm_time.gather(r, root=0)
        tmp = None

        if self.comm_time.Get_rank() == 0:
            r_full = np.vstack(r_full)
            tmp = la.norm(r_full)

        tmp = self.comm_time.bcast(tmp, root=0)
        self.conv[it] = tmp

    def forward_solve(self, lvl):
        """
        """
        if self.comm_time.Get_rank() == 0:
            for i in range(1, np.size(self.t[lvl])):
                self.u[lvl][i] = self.g[lvl][i] + self.phi(self.u[lvl][i - 1], self.t[lvl][i - 1], self.t[lvl][i],
                                                           self.app[lvl])

    def spatial_coarsening_per_level(self, spatial_coarsening, spatial_max_level):
        self.spatial_coarsening = [False] * self.lvl_max
        spatial_level = 0
        for lvl in range(self.lvl_max):
            if spatial_coarsening and spatial_level < spatial_max_level - 1 and self.lvl_max - lvl <= spatial_max_level:
                self.spatial_coarsening[lvl] = True
                spatial_level += 1
            else:
                self.spatial_coarsening[lvl] = False

    def fas_residual(self, lvl):
        """
        """
        if self.spatial_coarsening[lvl]:
            for i in range(len(self.proc_data[lvl]['all_cpts'])):
                self.v[lvl + 1][i]=self.restriction(np.copy(self.u[lvl][self.proc_data[lvl]['all_cpts'][i]]),
                                               self.app[lvl]['trans'])
            #self.v[lvl + 1] = self.restriction(np.copy(self.u[lvl][self.proc_data[lvl]['all_cpts']]),
            #                                   self.app[lvl]['trans'])
            if np.size(self.proc_data[lvl]['cpts']) > 0:
                for i in range(0, np.size(self.proc_data[lvl]['cpts'])):
                    j = i + self.proc_data[lvl]['first_i']
                    if j != 0:
                        self.g[lvl + 1][i, :] = self.restriction(
                            self.g[lvl][self.proc_data[lvl]['cpts'][i]] - self.u[lvl][
                                self.proc_data[lvl]['cpts'][i]] + self.phi(
                                self.u[lvl][self.proc_data[lvl]['cpts'][i] - 1],
                                self.t[lvl][self.proc_data[lvl]['cpts'][i] - 1],
                                self.t[lvl][self.proc_data[lvl]['cpts'][i]],
                                self.app[lvl]),
                            self.app[lvl]['trans']) + self.v[lvl + 1][
                                                    j] - self.phi(self.v[lvl + 1][j - 1], self.t[lvl + 1][j - 1],
                                                                  self.t[lvl + 1][j], self.app[lvl + 1])
        else:
            self.v[lvl + 1] = np.copy(self.u[lvl][self.proc_data[lvl]['all_cpts']])
            if np.size(self.proc_data[lvl]['cpts']) > 0:
                for i in range(0, np.size(self.proc_data[lvl]['cpts'])):
                    j = i + self.proc_data[lvl]['first_i']
                    if j != 0:
                        self.g[lvl + 1][i] = self.g[lvl][self.proc_data[lvl]['cpts'][i]] \
                                             + self.phi(self.u[lvl][self.proc_data[lvl]['cpts'][i] - 1],
                                                        self.t[lvl][self.proc_data[lvl]['cpts'][i] - 1],
                                                        self.t[lvl][self.proc_data[lvl]['cpts'][i]],
                                                        self.app[lvl]) - self.u[lvl][self.proc_data[lvl]['cpts'][i]] \
                                             - self.phi(self.v[lvl + 1][j - 1], self.t[lvl + 1][j - 1],
                                                        self.t[lvl + 1][j], self.app[lvl + 1]) + \
                                             self.v[lvl + 1][j]

        self.g[lvl + 1] = np.vstack(
            self.comm_time.allgather(self.g[lvl + 1][0:np.size(self.proc_data[lvl]['cpts']), :]))
        self.u[lvl + 1] = np.copy(self.v[lvl + 1])


    def nested_iteration(self):
        self.forward_solve(self.lvl_max - 1)
        self.u[-1] = self.comm_time.bcast(self.u[-1], root=0)

        for lvl in range(self.lvl_max - 2, -1, -1):
            if self.spatial_coarsening[lvl]:
                for i in range(np.size(self.u[lvl + 1],0)):
                    self.u[lvl][::self.m[lvl]][i]= self.interpolation(self.u[lvl + 1][i], self.app[lvl]['trans'])
                #self.u[lvl][::self.m[lvl]] = self.interpolation(self.u[lvl + 1], self.app[lvl]['trans'])
            else:
                self.u[lvl][::self.m[lvl]] = self.u[lvl + 1]
            if lvl > 0:
                self.iteration(lvl, 1, 'V', 0, True)

    def exchange(self, lvl):
        self.u[lvl] = np.vstack(self.comm_time.allgather(
            self.u[lvl][self.proc_data[lvl]['all_pts'][0]:(self.proc_data[lvl]['all_pts'][-1] + 1), :] if np.size(
                self.proc_data[lvl]['all_pts']) > 0 else np.empty(
                (0, self.u[lvl].shape[1]))))

    def f_exchange(self, lvl):
        rank = self.comm_time.Get_rank()
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][0] > 0 and
                np.size(np.where(self.proc_data[lvl]['cpts'] == self.proc_data[lvl]['all_pts'][0])[0]) == 1):
            self.comm_time.Recv(self.u[lvl][self.proc_data[lvl]['all_pts'][0] - 1], rank - 1,
                                tag=self.proc_data[lvl]['all_pts'][0] - 1)
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][-1] < np.size(self.t[lvl]) - 1 and
                np.size(np.where(self.proc_data[lvl]['fpts'] == self.proc_data[lvl]['all_pts'][-1])[0]) == 1 and
                np.size(np.where(self.proc_data[lvl]['all_cpts'] == self.proc_data[lvl]['all_pts'][-1] + 1)[0]) == 1):
            self.comm_time.Send(self.u[lvl][self.proc_data[lvl]['all_pts'][-1]], rank + 1,
                                tag=self.proc_data[lvl]['all_pts'][-1])

    def c_exchange(self, lvl):
        rank = self.comm_time.Get_rank()
        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][0] > 0 and
                np.size(np.where(self.proc_data[lvl]['fpts'] == self.proc_data[lvl]['all_pts'][0])[0]) == 1 and
                np.size(np.where(self.proc_data[lvl]['all_cpts'] == self.proc_data[lvl]['all_pts'][0] - 1)[0]) == 1):
            self.comm_time.Recv(self.u[lvl][self.proc_data[lvl]['all_pts'][0] - 1], rank - 1,
                                tag=self.proc_data[lvl]['all_pts'][0] - 1)

        if (self.proc_data[lvl]['all_pts'].shape[0] > 0 and
                self.proc_data[lvl]['all_pts'][-1] < np.size(self.t[lvl]) - 1 and
                np.size(np.where(self.proc_data[lvl]['cpts'] == self.proc_data[lvl]['all_pts'][-1])[0]) == 1):
            self.comm_time.Send(self.u[lvl][self.proc_data[lvl]['all_pts'][-1]], rank + 1,
                                tag=self.proc_data[lvl]['all_pts'][-1])

    def levels(self):
        self.u = [np.array(0)] * self.lvl_max
        self.v = [np.array(0)] * self.lvl_max
        self.g = [np.array(0)] * self.lvl_max
        self.t = [np.array(0)] * self.lvl_max
        self.proc_data = [{}] * self.lvl_max

        i = 0
        for lvl in range(self.lvl_max):
            self.t[lvl] = self.problem.t[np.array(range(0, np.size(self.problem.t), self.m[lvl] ** lvl))]
            self.u[lvl] = np.zeros((np.size(self.t[lvl]), self.problem.nx[i]))
            if lvl != 0:
                self.v[lvl] = np.zeros_like(self.u[lvl])
            self.g[lvl] = np.zeros_like(self.u[lvl])
            self.proc_data[lvl] = self.setup_points(nt=np.size(self.t[lvl]), m=self.m[lvl],
                                                    size=self.comm_time.Get_size(), rank=self.comm_time.Get_rank())
            if self.spatial_coarsening[lvl]:
                i = i + 1

    def setup(self, lvl_max, m, it, tol, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0):

        if self.comm_time.Get_rank() == 0:
            logging.info("Start setup")

        runtime_setup_start = time.time()
        self.lvl_max = lvl_max
        self.it = it
        self.tol = tol

        if isinstance(m, (list, int, np.ndarray)):
            if type(m) == int:
                self.m = np.ones(lvl_max, dtype=int) * m
            elif type(m) == list:
                self.m = np.array(m, dtype=int)
        else:
            raise Exception('unknown type for m')

        self.spatial_coarsening_per_level(spatial_coarsening=spatial_coarsening, spatial_max_level=spatial_max_lvl)
        self.levels()

        self.app = self.problem.setup(lvl_max=self.lvl_max, t=self.t, spatial_coarsening=self.spatial_coarsening)

        self.u[0][0] = self.problem.initial_value()
        if nested_iteration:
            self.nested_iteration()

        self.conv = np.zeros(it + 1)
        self.convergence_criteria(it=0)

        self.setup_done = True

        runtime_setup_stop = time.time()

        if self.comm_time.Get_rank() == 0:
            logging.info(f"Setup took {runtime_setup_stop - runtime_setup_start} s")
            logging.info(f"step 0 | norm r= {self.conv[0]}")

    def solve(self, cf_iter=1, cycle_type='V'):
        """

        :rtype: object
        """
        if not self.setup_done:
            raise Exception('setup solver')

        if self.comm_time.Get_rank() == 0:
            logging.info("Start solve")

        runtime_solve_start = time.time()
        for iteration in range(self.it):

            time_it_start = time.time()
            self.iteration(lvl=0, cf_iter=cf_iter, cycle_type=cycle_type, iteration=iteration, first_f=True)
            time_it_stop = time.time()
            self.convergence_criteria(it=iteration + 1)

            if self.comm_time.Get_rank() == 0:
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
            self.save()

        return self.u[0]

    def save(self):
        path = 'results/' + self.problem.info() + 'L-' + str(self.lvl_max) + '|m-' + str(self.m) + '|sC-' + str(
            self.spatial_coarsening)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        np.savez(path + '/res', u=self.u[0], conv=self.conv, t=self.t[0], runtime=self.runtime_solve)

    def error_correction(self, lvl):
        e = self.u[lvl + 1][
            self.proc_data[lvl]['first_i']:self.proc_data[lvl]['first_i'] + self.proc_data[lvl]['block_size']] - \
            self.v[lvl + 1][
            self.proc_data[lvl]['first_i']:self.proc_data[lvl]['first_i'] + self.proc_data[lvl]['block_size']]


        if self.spatial_coarsening[lvl]:
            tmp = np.zeros((self.proc_data[lvl]['block_size'], np.size(self.u[lvl],1)))
            for i in range(self.proc_data[lvl]['block_size']):
                tmp[i] = self.interpolation(e[i], self.app[lvl]['trans'])
            e = tmp

            #e = self.interpolation(e, self.app[lvl]['trans'])

        self.u[lvl][self.proc_data[lvl]['cpts']] += e

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
