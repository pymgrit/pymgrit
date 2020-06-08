"""
MGRIT optimized for the induction
machine model "im_3kW". (https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines)
"""

import numpy as np

from pymgrit.core.mgrit import Mgrit


class MgritMachine(Mgrit):
    """
    MGRIT optimized for the getdp induction machine
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        MGRIT optimized for the getdp induction machine
        """
        super(MgritMachine, self).__init__(*args, **kwargs)

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
