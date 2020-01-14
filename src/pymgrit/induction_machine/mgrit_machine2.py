"""
MGRIT optimized for the GETDP induction machine
with residual norm as convergence criteria
"""

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
