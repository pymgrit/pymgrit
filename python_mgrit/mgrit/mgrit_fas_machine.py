from mgrit import mgrit_fas
import numpy as np
import logging
import time
import pathlib
import datetime


class MgritFasMachine(mgrit_fas.MgritFas):

    def __init__(self, compute_f_after_convergence: bool, *args, **kwargs) -> None:
        """
        MGRIT optimized for the GETDP induction machine
        :param compute_f_after_convergence:
        :param args:
        :param kwargs:
        """
        super(MgritFasMachine, self).__init__(*args, **kwargs)
        self.last_it = []
        self.compute_f_after_convergence = compute_f_after_convergence

    def nested_iteration(self) -> None:
        """
        Generate initial approximation by the computation and interpolation of approximations on coarser grids
        Performs the nested_iteration with a continuous signal
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

    def convergence_criteria(self, it: int) -> None:
        """
        Maximum norm of all C-points
        :param it: Iteration number
        """
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
        tmp_output_fcn = self.output_fcn
        self.output_fcn = None
        super(MgritFasMachine, self).solve()
        self.output_fcn = tmp_output_fcn
        if self.compute_f_after_convergence:
            if self.comm_time.Get_rank() == 0:
                logging.info("Start post-processing: F-relax")
            runtime_pp_start = time.time()
            self.f_relax(lvl=0)
            runtime_pp_stop = time.time()
            if self.comm_time.Get_rank() == 0:
                logging.info(f"Post-processing took {runtime_pp_stop - runtime_pp_start} s")
        self.last_it = np.zeros_like(self.last_it)
        if self.output_fcn is not None:
            self.output_fcn(self)
        return {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
                't': self.problem[0].t, 'time_setup': self.runtime_setup}
