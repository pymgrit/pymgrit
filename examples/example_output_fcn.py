"""
Save and plot the MGRIT approximation of the solution after each iteration
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

from mpi4py import MPI

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution; here, we include the iteration number in the path name
        path = 'results/' + 'brusselator' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Save solution to file.
        # Useful member variables of MGRIT solver:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        #   - self.comm_time_rank : Time communicator rank
        np.save(path + '/brusselator-rank' + str(self.comm_time_rank),
                [[[self.t[0][i], self.u[0][i]] for i in self.index_local[0]]])  # Solution and time at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0)

    # Solve Brusselator system
    info = mgrit.solve()

    # Plot the MGRIT approximation of the solution after each iteration
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Dynamic images
        iterations_needed = len(info['conv']) + 1
        cols = 2
        rows = iterations_needed // cols + iterations_needed % cols
        position = range(1, iterations_needed + 1)
        fig = plt.figure(1, figsize=[10, 10])
        for i in range(iterations_needed):
            # Load each file and add the loaded values to sol
            sol = []
            path = 'results/brusselator/' + str(i)
            for filename in os.listdir(path):
                data = np.load(path + '/' + filename, allow_pickle=True).tolist()[0]
                sol += data
            # Sort the solution list by the time
            sol.sort(key=lambda tup: tup[0])
            # Get the solution values
            values = np.array([i[1].get_values() for i in sol])
            ax = fig.add_subplot(rows, cols, position[i])
            # Plot the two solution values at each time point
            ax.scatter(values[:, 0], values[:, 1])
            ax.set(xlabel='x', ylabel='y')
        fig.tight_layout(pad=2.0)
        plt.show()


if __name__ == '__main__':
    main()
