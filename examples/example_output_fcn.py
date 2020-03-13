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

        # Save solution to file; here, we have two solution values at each time point.
        # Useful member variables of MGRIT solver:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        np.save(path + '/brusselator-' + str(self.comm_time_rank),
                [self.u[0][i].get_values() for i in self.index_local[0]])  # Solution values at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0)

    # Solve Brusselator system
    info = mgrit.solve()

    # Plot the MGRIT approximation of the solution after each iteration
    iterations_needed = len(info['conv']) + 1
    cols = 2
    rows = iterations_needed // cols + iterations_needed % cols
    position = range(1, iterations_needed + 1)
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        fig = plt.figure(1, figsize=[10, 10])
        for i in range(iterations_needed):
            files = []
            path = 'results/brusselator/' + str(i) + '/'
            # Construct solution from multiple files
            for filename in os.listdir(path):
                files.append([int(filename[filename.find('-') + 1: -4]), np.load(path + filename, allow_pickle=True)])
            files.sort(key=lambda tup: tup[0])
            sol = np.vstack([l.pop(1) for l in files])
            ax = fig.add_subplot(rows, cols, position[i])
            # Plot the two solution values at each time point
            ax.scatter(sol[:, 0], sol[:, 1])
            ax.set(xlabel='x', ylabel='y')
        fig.tight_layout(pad=2.0)
        plt.show()


if __name__ == '__main__':
    main()
