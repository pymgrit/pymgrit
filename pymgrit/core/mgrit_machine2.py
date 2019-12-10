"""
MGRIT optimized for the GETDP induction machine
with residual norm as convergence criteria
"""

import logging
import time
import numpy as np

from pymgrit.core import mgrit


class MgritMachine2(mgrit.Mgrit):
    """
    MGRIT optimized for the GETDP induction machine
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        MGRIT optimized for the GETDP induction machine
        :param compute_f_after_convergence:
        :param args:
        :param kwargs:
        """
        super(MgritMachine2, self).__init__(*args, **kwargs)

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

    def f_relax(self, lvl: int) -> None:
        """
        :param lvl: the corresponding MGRIT level
        """
        runtime_f = time.time()
        tmp_send = False
        req_s = None
        rank = self.comm_time_rank
        if self.index_local_f[lvl].size > 0:
            for i in np.nditer(self.index_local_f[lvl]):
                try:
                    if self.comm_front[lvl] and i == np.min(self.index_local_f[lvl]):
                        self.u[lvl][0] = self.comm_time.recv(source=self.get_from[lvl], tag=rank)
                    if lvl == 0:
                        self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                        t_start=self.t[lvl][i - 1],
                                                        t_stop=self.t[lvl][i])
                    else:
                        self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                         t_start=self.t[lvl][i - 1],
                                                                         t_stop=self.t[lvl][i])
                    if self.comm_back[lvl] and i == np.max(self.index_local_f[lvl]):
                        tmp_send = True
                        req_s = self.comm_time.isend(self.u[lvl][-1], dest=self.send_to[lvl], tag=self.send_to[lvl])
                except:
                    print(i)
                    raise Exception('error at point'+str(i))

        if tmp_send:
            req_s.wait()

        logging.debug(f"F-relax on {rank} took {time.time() - runtime_f} s")
