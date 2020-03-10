"""
Access and plot the solution
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():
    # Define output function that writes the solution to a file
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'dahlquist'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Save solution to file; here, we just have a single solution value at each time point.
        # Use local time interval to distinguish between processes:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        np.save(path + '/dahlquist',  # Local time interval
                [self.u[0][i].get_values() for i in self.index_local[0]])   # Solution values at local time points

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the output function
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, output_fcn=output_fcn)

    # Solve the test problem
    info = mgrit.solve()

    # Plot the solution (Note: modifications necessary if more than one process is used for the simulation!)
    t = np.linspace(dahlquist.t_start, dahlquist.t_end, dahlquist.nt)
    sol = np.load('results/dahlquist/dahlquist.npy')
    plt.plot(t, sol)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.show()


if __name__ == '__main__':
    main()
