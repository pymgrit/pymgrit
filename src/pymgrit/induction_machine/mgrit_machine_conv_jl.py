"""
MGRIT optimized for the induction
machine model "im_3kW". (https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines)
Using joule losses as stopping criteria
"""

import logging
import time
import numpy as np

from pymgrit.core.mgrit import Mgrit


class MgritMachineConvJl(Mgrit):
    """
    MGRIT optimized for the getdp induction machine
    """

    def __init__(self, compute_f_after_convergence: bool, *args, **kwargs) -> None:
        """
        MGRIT optimized for the getdp induction machine

        :param compute_f_after_convergence: computes solution of F-points at the end
        """
        super(MgritMachineConvJl, self).__init__(*args, **kwargs)
        self.last_it = []
        self.compute_f_after_convergence = compute_f_after_convergence
        self.convergence_criteria(iteration=0)

    def nested_iteration(self) -> None:
        """
        Generates an initial approximation on the finest grid
        by solving the problem on the coarsest grid and interpolating
        the approximation to the finest level.

        Performs nested iterations with a sin rhs
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

    def iteration(self, lvl: int, cycle_type: str, iteration: int, first_f: bool):
        """
        MGRIT iteration on level lvl

        :param lvl: MGRIT level
        :param cycle_type: Cycle type
        :param iteration: Number of current iteration
        :param first_f: F-relaxation at the beginning
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for _ in range(self.cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        if lvl > 0:
            self.f_relax(lvl=lvl)

        if lvl != 0 and cycle_type == 'F':
            self.f_exchange(lvl=lvl)
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def convergence_criteria(self, iteration: int) -> None:
        """
        Maximum norm of all C-points

        :param iteration: iteration number
        """
        if len(self.last_it) != len(self.index_local_c[0]):
            self.last_it = np.zeros(len(self.index_local_c[0]))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        if self.index_local_c[0].size > 0:
            for i in np.nditer(self.index_local_c[0]):
                new[j] = self.u[0][i].jl
                j = j + 1
            tmp = 100 * np.max(
                np.abs(np.abs(np.divide((new - self.last_it), new, out=np.zeros_like(self.last_it), where=new != 0))))

        tmp = self.comm_time.allgather(tmp)
        self.conv[iteration] = np.max(np.abs(tmp))
        self.last_it = np.copy(new)

    def solve(self) -> dict:
        """
        Driver function for solving the problem using MGRIT.

        Performs MGRIT iterations until a stopping criterion is fulfilled or
        the maximum number of iterations is reached.

        Post operation for computing solution at F-points

        :return: dictionary with residual history, setup time, and solve time
        """
        tmp_output_fcn = self.output_fcn
        self.output_fcn = None
        super(MgritMachineConvJl, self).solve()
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
        return {'conv': self.conv[np.where(self.conv != 0)], 'time_setup': self.runtime_setup,
                'time_solve': self.runtime_solve}
