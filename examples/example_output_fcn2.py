"""
Access and plot the solution
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'brusselator' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have two solution values at each time point.
        np.save(path + '/brusselator',
                [self.u[0][i].get_values() for i in self.index_local[0]])  # Solution values at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0,
                  nested_iteration=False)

    # Solve Brusselator system
    info = mgrit.solve()

    Tot = len(info['conv']) + 1
    Cols = 2

    Rows = Tot // Cols
    Rows += Tot % Cols

    Position = range(1, Tot + 1)

    fig = plt.figure(1, figsize=[10,10])
    # Plot the solution (Note: modifications necessary if more than one process is used for the simulation!)
    for i in range(Tot):
        res = np.load('results/brusselator/' + str(i) + '/brusselator.npy', allow_pickle=True)
        ax = fig.add_subplot(Rows, Cols, Position[i])
        ax.scatter(res[:,0], res[:,1])
        plt.xlabel('t')
        plt.ylabel('u(t)')
    plt.show()


if __name__ == '__main__':
    main()
