"""
Apply two-level MGRIT with FCF-relaxation to solve Brusselator system,
save the MGRIT approximation of the solution at the end of the simulation,
plot the MGRIT approximation
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'brusselator'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have two solution values at each time point.
        np.save(path + '/brusselator',
                [self.u[0][i] for i in self.index_local[0]])  # Solution values at local time points

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], cf_iter=1, output_fcn=output_fcn)

    # Solve Brusselator system
    infos = mgrit.solve()

    # Load MGRIT approximation of solution
    sol = np.load('results/brusselator/brusselator.npy', allow_pickle=True)

    # Plot solution using member function of class VectorBrusselator
    for i in range(0, len(sol)):
        sol[i].plotSolution()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
