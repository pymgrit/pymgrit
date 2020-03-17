"""
Example to customize the MGRIT algorithm by a different
convergence criteria.
"""

import numpy as np

from pymgrit.core.mgrit import Mgrit
from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit


class MgritCustomized(Mgrit):
    """
    Customized MGRIT with maximum norm of the relative
    difference of two successive iterates as convergence criteria
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Cumstomized MGRIT constructor
        :param args:
        :param kwargs:
        """
        # Call parent constructor
        super(MgritCustomized, self).__init__(*args, **kwargs)
        # New member variable for saving the C-points values of the last iteration
        self.last_it = []
        # Initialize the new member variable
        self.convergence_criteria(iteration=0)

    def convergence_criteria(self, iteration: int) -> None:
        """
        Stops if the maximum norm of the relative
        difference of two successive iterates
        at C-points is below the stopping tolerance.
        :param iteration: Iteration number
        """

        # Create structure on the first function call
        if len(self.last_it) != len(self.index_local_c[0]):
            self.last_it = np.zeros((len(self.index_local_c[0]), len(self.u[0][0].get_values())))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        # If process has a C-point
        if self.index_local_c[0].size > 0:
            # Loop over all C-points of the process
            for i in np.nditer(self.index_local_c[0]):
                new[j] = self.u[0][i].get_values()
                j = j + 1
            # Compute relative difference between two iterates
            tmp = 100 * np.max(
                np.abs(np.abs(np.divide((new - self.last_it), new, out=np.zeros_like(self.last_it), where=new != 0))))

        # Communicate the local value
        tmp = self.comm_time.allgather(tmp)
        # Maximum norm
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
