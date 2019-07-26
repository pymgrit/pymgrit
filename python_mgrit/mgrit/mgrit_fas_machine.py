from mgrit import mgrit_fas
import numpy as np
from scipy import linalg as la
import logging
import time


class MgritFasMachine(mgrit_fas.MgritFas):

    def __init__(self, compute_f_after_convergence, *args, **kwargs):
        super(MgritFasMachine, self).__init__(*args, **kwargs)
        self.last_it = []
        self.compute_f_after_convergence = compute_f_after_convergence

    def nested_iteration(self) -> None:
        """

        """
        change = False
        tmp_problem_pwm = np.zeros(len(self.problem))
        if self.problem[0].pwm:
            change = True
            for lvl in range(len(self.problem)):
                tmp_problem_pwm[lvl] = self.problem[lvl].pwm
                self.problem[lvl].fopt[-1] = 0

        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl, 'V', 0, True)

        if change:
            for lvl in range(len(self.problem)):
                self.problem[lvl].fopt[-1] = tmp_problem_pwm[lvl]

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
        if len(self.last_it) != len(self.index_local_c[0]):
            self.last_it = np.zeros(len(self.index_local_c[0]))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        if len(self.index_local_c[0]) > 0:
            for i in np.nditer(self.index_local_c[0]):
                new[j] = self.u[0][i].jl
                j = j + 1
            tmp = np.max(np.abs(new - self.last_it))

        tmp = self.comm_time.allgather(tmp)
        self.conv[it] = np.max(np.abs(tmp))
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
        solution = self.comm_time.gather([self.u[0][i] for i in self.index_local[0]], root=0)
        if self.comm_time.Get_rank() == 0:
            solution = [item for sublist in solution for item in sublist]
        self.last_it = np.zeros_like(self.last_it)
        return {'u': solution, 'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t}
