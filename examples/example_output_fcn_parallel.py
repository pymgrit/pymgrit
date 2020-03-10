"""
Access and plot the solution
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
        # Set path to solution
        path = 'results/' + 'brusselator_parallel' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have two solution values at each time point.
        np.save(path + '/brusselator-' + str(self.comm_time_rank),
                [self.u[0][i].get_values() for i in self.index_local[0]])  # Solution values at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0,
                  nested_iteration=False)

    # Solve Brusselator system
    info = mgrit.solve()

    iterations_needed = len(info['conv']) + 1
    cols = 2
    rows = iterations_needed // cols + iterations_needed % cols
    position = range(1, iterations_needed + 1)
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        fig = plt.figure(1, figsize=[10, 10])
        for i in range(iterations_needed):
            files = []
            path = 'results/brusselator_parallel/'+str(i)+'/'
            for filename in os.listdir(path):
                files.append([int(filename[filename.find('-') + 1: -4]), np.load(path + filename, allow_pickle=True)])
            files.sort(key=lambda tup: tup[0])
            res = np.vstack([l.pop(1) for l in files])
            ax = fig.add_subplot(rows, cols, position[i])
            ax.scatter(res[:, 0], res[:, 1])
        plt.show()

if __name__ == '__main__':
    main()
