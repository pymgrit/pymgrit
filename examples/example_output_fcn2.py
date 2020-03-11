"""
Save and plot the MGRIT approximation of the solution after each iteration

Note: This example assumes a sequential run of the simulation.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution (one path for each iteration)
        path = 'results/' + 'brusselator' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have two solution values at each time point.
        # Useful member variables of MGRIT solver:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        np.save(path + '/brusselator',
                [self.u[0][i].get_values() for i in self.index_local[0]])  # Solution values at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0,
                  nested_iteration=True)

    # Solve Brusselator system
    info = mgrit.solve()

    # Plot the MGRIT approximation of the solution after each iteration
    iterations_needed = len(info['conv']) + 1
    cols = 2
    rows = iterations_needed // cols + iterations_needed % cols
    position = range(1, iterations_needed + 1)
    fig = plt.figure(1, figsize=[10, 10])
    for i in range(iterations_needed):
        sol = np.load('results/brusselator/' + str(i) + '/brusselator.npy', allow_pickle=True)
        ax = fig.add_subplot(rows, cols, position[i])
        ax.scatter(sol[:, 0], sol[:, 1])
        ax.set_title('iteration ' + str(i))
        ax.set(xlabel='x', ylabel='y')
    fig.tight_layout(pad=2.0)
    plt.show()


if __name__ == '__main__':
    main()
