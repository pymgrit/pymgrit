"""

"""

import numpy as np

from pymgrit.core.mgrit import Mgrit
from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit


class MgritCustomized(Mgrit):
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
        super(MgritCustomized, self).__init__(*args, **kwargs)
        self.last_it = []
        self.convergence_criteria(iteration=0)

    def convergence_criteria(self, iteration: int) -> None:
        """
        Stops if the maximum norm of the relative
        difference of two successive iterates
        at C-points is below the stopping tolerance.
        :param iteration: Iteration number
        """
        if len(self.last_it) != len(self.index_local_c[0]):
            self.last_it = np.zeros((len(self.index_local_c[0]), len(self.u[0][0].get_values())))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        if self.index_local_c[0].size > 0:
            for i in np.nditer(self.index_local_c[0]):
                new[j] = self.u[0][i].get_values()
                j = j + 1
            tmp = 100 * np.max(
                np.abs(np.abs(np.divide((new - self.last_it), new, out=np.zeros_like(self.last_it), where=new != 0))))

        tmp = self.comm_time.allgather(tmp)
        self.conv[iteration] = np.max(np.abs(tmp))
        self.last_it = np.copy(new)


def main():
    # Create two-level time-grid hierarchy for the ODE system describing Arenstorf orbits
    ahrenstorf_lvl_0 = ArenstorfOrbit(t_start=0, t_stop=17.06521656015796, nt=10001)
    ahrenstorf_lvl_1 = ArenstorfOrbit(t_interval=ahrenstorf_lvl_0.t[::100])

    # Use the customized MGRIT algorithm to solve the problem.
    # Stopps if the maximum relative change in all four variables of arenstorf orbit is smaller than 1% for all C-points
    info = MgritCustomized(problem=[ahrenstorf_lvl_0, ahrenstorf_lvl_1], tol=1).solve()


if __name__ == '__main__':
    main()
