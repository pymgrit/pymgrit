"""
Customized MGRIT solver that uses a user-defined stopping criterion.

Apply customized two-level MGRIT with FCF-relaxation to compute an Arenstorf orbit.
"""

import numpy as np

from pymgrit.core.mgrit import Mgrit
from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit


class MgritCustomized(Mgrit):
    """
    Customized MGRIT class.

    Use maximum norm of the relative difference at C-points of two successive
    iterates as convergence criterion.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Cumstomized MGRIT constructor.
        """
        # Call parent constructor
        super(MgritCustomized, self).__init__(*args, **kwargs)
        # New member variable for saving the C-point values of the last iteration
        self.last_it = []
        # Initialize the new member variable
        self.convergence_criterion(iteration=0)

    def convergence_criterion(self, iteration: int) -> None:
        """
        Stopping criterion based on achieving a maximum relative difference at C-points
        of two successive iterates below the specified stopping tolerance.
        Note: The stopping tolerance is specified when setting up the solver.

        :param iteration: Iteration number
        """

        # Create list in the first function call
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
        # Take maximum norm
        self.conv[iteration] = np.max(np.abs(tmp))
        self.last_it = np.copy(new)


def main():
    # Create two-level time-grid hierarchy for the ODE system describing Arenstorf orbits
    ahrenstorf_lvl_0 = ArenstorfOrbit(t_start=0, t_stop=17.06521656015796, nt=10001)
    ahrenstorf_lvl_1 = ArenstorfOrbit(t_interval=ahrenstorf_lvl_0.t[::100])

    # Set up customized MGRIT solver and solve the problem.
    # Note: Setting the solver tolerance to 1 means that iterations stop
    #       if the maximum relative change at C-points of all four variables of the ODE system
    #       is smaller than 1%.
    info = MgritCustomized(problem=[ahrenstorf_lvl_0, ahrenstorf_lvl_1], tol=1).solve()


if __name__ == '__main__':
    main()
