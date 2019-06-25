from scipy import linalg as la
import time
from operator import itemgetter
from itertools import *
import numpy as np


class MgritFasS:
    """
    """

    def __init__(self, problem):
        """
        """
        self.problem = problem
        self.problem = problem
        self.phi = self.problem.phi
        self.restriction = self.problem.restriction
        self.interpolation = self.problem.interpolation

        # dummies
        self.u = []
        self.v = []
        self.g = []
        self.t = []
        self.core_data = []
        self.spatial_coarsening = []
        self.m = []
        self.it = 0
        self.tol = 0
        self.res = []
        self.lvl_max = 0
        self.app = []
        self.setup_done = False

    def c_relax(self, lvl):
        """
        """
        if len(self.core_data[lvl]['cpts']) > 0:
            for i in np.nditer(self.core_data[lvl]['cpts']):
                if i != 0:
                    self.u[lvl][i] = self.g[lvl][i] + self.phi(self.u[lvl][i - 1], self.t[lvl][i - 1], self.t[lvl][i],
                                                               self.app[lvl])

    def f_relax(self, lvl):
        """
        """
        if len(self.core_data[lvl]['fpts']) > 0:
            for i in np.nditer(self.core_data[lvl]['fpts']):
                if i != 0:
                    self.u[lvl][i] = self.g[lvl][i] + self.phi(self.u[lvl][i - 1], self.t[lvl][i - 1], self.t[lvl][i],
                                                               self.app[lvl])

    def convergence_criteria(self, it):
        """
        """
        cpts = np.array(range(0, len(self.t[0]), self.m[0]))
        r = np.zeros_like(self.u[0])
        for i in range(1, np.size(self.t[0])):
            if i in cpts:
                r[i] = self.phi(self.u[0][i - 1], self.t[0][i - 1], self.t[0][i], self.app[0]) - self.u[0][i]
        self.res[it] = la.norm(r)

    def forward_solve(self, lvl):
        """
        """
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

    def nested_iteration(self):
        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            if self.spatial_coarsening[lvl]:
                self.u[lvl][::self.m[lvl]] = self.interpolation(self.u[lvl + 1], self.app[lvl]['trans'])
            else:
                self.u[lvl][::self.m[lvl]] = self.u[lvl + 1]
            if lvl > 0:
                self.iteration(lvl, 1, 'V', 0, True)

    def fas_residual(self, lvl):
        """
        """
        if self.spatial_coarsening[lvl]:
            self.v[lvl + 1] = self.restriction(np.copy(self.u[lvl][self.core_data[lvl]['all_cpts']]),
                                               self.app[lvl]['trans'])
            if np.size(self.core_data[lvl]['cpts']) > 0:
                for i in range(0, np.size(self.core_data[lvl]['cpts'])):
                    j = i + self.core_data[lvl]['first_i']
                    if j != 0:
                        self.g[lvl + 1][i, :] = self.restriction(
                            self.g[lvl][self.core_data[lvl]['cpts'][i]] - self.u[lvl][
                                self.core_data[lvl]['cpts'][i]] + self.phi(
                                self.u[lvl][self.core_data[lvl]['cpts'][i] - 1],
                                self.t[lvl][self.core_data[lvl]['cpts'][i] - 1],
                                self.t[lvl][self.core_data[lvl]['cpts'][i]],
                                self.app[lvl]),
                            self.app[lvl]['trans']) + self.v[lvl + 1][
                                                    j] - self.phi(self.v[lvl + 1][j - 1], self.t[lvl + 1][j - 1],
                                                                  self.t[lvl + 1][j], self.app[lvl + 1])
        else:
            self.v[lvl + 1] = np.copy(self.u[lvl][self.core_data[lvl]['all_cpts']])
            if np.size(self.core_data[lvl]['cpts']) > 0:
                for i in range(0, np.size(self.core_data[lvl]['cpts'])):
                    j = i + self.core_data[lvl]['first_i']
                    if j != 0:
                        self.g[lvl + 1][i] = self.g[lvl][self.core_data[lvl]['cpts'][i]] \
                                             + self.phi(self.u[lvl][self.core_data[lvl]['cpts'][i] - 1],
                                                        self.t[lvl][self.core_data[lvl]['cpts'][i] - 1],
                                                        self.t[lvl][self.core_data[lvl]['cpts'][i]],
                                                        self.app[lvl]) - self.u[lvl][self.core_data[lvl]['cpts'][i]] \
                                             - self.phi(self.v[lvl + 1][j - 1], self.t[lvl + 1][j - 1],
                                                        self.t[lvl + 1][j], self.app[lvl + 1]) + \
                                             self.v[lvl + 1][j]

    def setup(self, lvl_max, m, it, tol, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0):
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

        self.res = np.zeros(it + 1)
        self.convergence_criteria(it=0)

        print("step ", 0, "| norm r=", self.res[0])

        self.setup_done = True

    def solve(self, cf_iter=1, cycle_type='V'):
        """

        :rtype: object
        """
        if not self.setup_done:
            raise Exception('setup solver')

        start = time.time()
        for iteration in range(self.it):

            self.iteration(lvl=0, cf_iter=cf_iter, cycle_type=cycle_type, iteration=iteration, first_f=True)
            self.convergence_criteria(it=iteration + 1)

            print("step ", iteration + 1, "| norm r=", self.res[iteration + 1])

            if self.res[iteration + 1] < self.tol:
                break

        end = time.time()
        print("time", end - start)
        return self.u

    def setup_and_solve(self, lvl_max, m, it_max, tol, cf_iter=1, cycle_type='V', nested_iteration=True,
                        spatial_coarsening=False, spatial_max_lvl=0):
        """
        """
        self.setup(lvl_max=lvl_max, m=m, it=it_max, tol=tol, nested_iteration=nested_iteration,
                   spatial_coarsening=spatial_coarsening, spatial_max_lvl=spatial_max_lvl)
        self.solve(cf_iter=cf_iter, cycle_type=cycle_type)
        return self.u

    def iteration(self, lvl, cf_iter, cycle_type, iteration, first_f):
        """
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)

        for i in range(cf_iter):
            self.c_relax(lvl=lvl)
            self.f_relax(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.u[lvl + 1] = np.copy(self.v[lvl + 1])

        self.iteration(lvl=lvl + 1, cf_iter=cf_iter, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.iteration(lvl=lvl, cf_iter=cf_iter, cycle_type='V', iteration=iteration, first_f=False)

    def error_correction(self, lvl):
        e = self.u[lvl + 1] - self.v[lvl + 1]

        if self.spatial_coarsening[lvl]:
            e = self.interpolation(e, self.app[lvl]['trans'])

        self.u[lvl][self.core_data[lvl]['cpts']] = self.u[lvl][self.core_data[lvl]['cpts']] + e

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

    def levels(self):
        self.u = [np.array(0)] * self.lvl_max
        self.v = [np.array(0)] * self.lvl_max
        self.g = [np.array(0)] * self.lvl_max
        self.t = [np.array(0)] * self.lvl_max
        self.core_data = [{}] * self.lvl_max

        i = 0
        for lvl in range(self.lvl_max):
            self.t[lvl] = self.problem.t[np.array(range(0, np.size(self.problem.t), self.m[lvl] ** lvl))]
            self.u[lvl] = np.zeros((np.size(self.t[lvl]), self.problem.nx[i]))
            if lvl != 0:
                self.v[lvl] = np.zeros_like(self.u[lvl])
            self.g[lvl] = np.zeros_like(self.u[lvl])
            self.core_data[lvl] = self.setup_points(nt=np.size(self.t[lvl]), m=self.m[lvl],
                                                    size=1, rank=0)
            if self.spatial_coarsening[lvl]:
                i = i + 1
