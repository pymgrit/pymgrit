from mgrit import mgrit_fas
import numpy as np
from scipy import linalg as la
import logging
import time


class MgritFasPointConvergence(mgrit_fas.MgritFas):

    def __init__(self, stop_unknown, compute_f_after_convergence, *args, **kwargs):
        super(MgritFasPointConvergence, self).__init__(*args, **kwargs)
        self.stop_unknown = stop_unknown
        self.compute_f_after_convergence = compute_f_after_convergence
        self.last_it = 0

    def iteration(self, lvl, cf_iter, cycle_type, iteration, first_f):
        """
        TODO
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            self.u[lvl] = self.comm_time.bcast(self.u[lvl], root=0)
            return

        if first_f:
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

        self.u[lvl + 1] = np.copy(self.v[lvl + 1])

        self.iteration(lvl=lvl + 1, cf_iter=cf_iter, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        if lvl > 0:
            self.f_relax(lvl=lvl)

        self.exchange(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.iteration(lvl=lvl, cf_iter=cf_iter, cycle_type='V', iteration=iteration, first_f=False)

    def convergence_criteria(self, it):
        cpts = np.array(range(0, len(self.t[0]), self.m[0]))
        self.conv[it] = la.norm(self.u[0][:, self.stop_unknown][cpts] - self.last_it)
        self.last_it = np.copy(self.u[0][:, self.stop_unknown][cpts])

    def solve(self, cf_iter=1, cycle_type='V'):
        super(MgritFasPointConvergence, self).solve(cf_iter=cf_iter, cycle_type=cycle_type)

        if self.compute_f_after_convergence:
            if self.comm_time.Get_rank() == 0:
                logging.info("Start post-processing: F-relax")
            runtime_pp_start = time.time()
            self.f_relax(lvl=0)
            self.exchange(lvl=0)
            runtime_pp_stop = time.time()
            if self.comm_time.Get_rank() == 0:
                logging.info(f"Post-processing took {runtime_pp_stop - runtime_pp_start} s")
                self.save()
        return self.u[0]
