from mpi4py import MPI
from mgrit import mgrit_fas
import numpy as np
from scipy import linalg as la
import logging
import time
import copy
import sys


class MgritFasMachine(mgrit_fas.MgritFas):

    def __init__(self, compute_f_after_convergence, problem, grid_transfer, it=100, tol=1e-7, nested_iteration=True, cf_iter=1, cycle_type='V',
                 comm_time=MPI.COMM_WORLD, comm_space=None, debug_lvl=logging.INFO):
        self.last_it = []
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


        if nested_iteration:
            if self.problem[0].pwm:
                tmp_problem_pwm = np.zeros(len(self.problem))
                for lvl in range(len(self.problem)):
                    tmp_problem_pwm[lvl] = self.problem[lvl].pwm
                    self.problem[lvl].fopt[-1] = 0
                self.nested_iteration()
                for lvl in range(len(self.problem)):
                    self.problem[lvl].fopt[-1] = tmp_problem_pwm[lvl]
            else:
                self.nested_iteration()

        runtime_setup_stop = time.time()

        if self.comm_time.Get_rank() == 0:
            logging.info(f"Setup took {runtime_setup_stop - runtime_setup_start} s")
        self.compute_f_after_convergence = compute_f_after_convergence


    def iteration(self, lvl, cycle_type, iteration, first_f):
        """
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if first_f:
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

        if lvl > 0:
            self.f_relax(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.f_exchange(lvl=lvl)
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def convergence_criteria(self, it):
        if len(self.last_it) != len(self.proc_data[0]['index_local_c']):
            self.last_it = np.zeros(len(self.proc_data[0]['index_local_c']))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        if len(self.proc_data[0]['index_local_c']) > 0:
            for i in np.nditer(self.proc_data[0]['index_local_c']):
                new[j] = self.u[0][i].jl
                j = j + 1
            tmp = la.norm(new - self.last_it)

        tmp = self.comm_time.allgather(tmp)
        for item in tmp:
            self.conv[it] += item ** 2

        self.conv[it] = self.conv[it] ** 0.5
        self.last_it = np.copy(new)

    def solve(self, cf_iter=1, cycle_type='V'):
        super(MgritFasMachine, self).solve()

        if self.compute_f_after_convergence:
            if self.comm_time.Get_rank() == 0:
                logging.info("Start post-processing: F-relax")
            runtime_pp_start = time.time()
            self.f_relax(lvl=0)
            runtime_pp_stop = time.time()
            if self.comm_time.Get_rank() == 0:
                logging.info(f"Post-processing took {runtime_pp_stop - runtime_pp_start} s")
        solution = self.comm_time.gather([self.u[0][i] for i in self.proc_data[0]['index_local']],root=0)
        if self.comm_time.Get_rank() == 0:
            solution = [item for sublist in solution for item in sublist]
        self.last_it = np.zeros_like(self.last_it)
        return {'u': solution, 'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t}
