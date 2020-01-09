"""
MGRIT for the parallel model, just one iteration
"""

from pymgrit.core import mgrit


class MgritParallelModel(mgrit.Mgrit):
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
        super(MgritParallelModel, self).__init__(*args, **kwargs)

    def convergence_criteria(self, iteration: int) -> None:
        """
        Stop after one iteration
        :param iteration: Iteration number
        """
        self.conv[iteration] = 1

    def iteration(self, lvl, cycle_type, iteration, first_f):
        """
        Performs one iteration
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
