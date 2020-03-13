"""
Apply two-level MGRIT with F-relaxation to compute an Arenstorf orbit,
save the MGRIT approximation of the solution at the end of the simulation,
plot the MGRIT approximation
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

from mpi4py import MPI

from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'arenstorf'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have four solution values at each time point.
        np.save(path + '/arenstorf-' + str(self.comm_time_rank),
                [self.u[0][i] for i in self.index_local[0]])  # Save solution values at local time points

    # Create two-level time-grid hierarchy for the ODE system describing Arenstorf orbits
    ahrenstorf_lvl_0 = ArenstorfOrbit(t_start=0, t_stop=17.06521656015796, nt=80001)
    ahrenstorf_lvl_1 = ArenstorfOrbit(t_interval=ahrenstorf_lvl_0.t[::320])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[ahrenstorf_lvl_0, ahrenstorf_lvl_1], cf_iter=0, output_fcn=output_fcn)

    # Compute Arenstorf orbit
    info = mgrit.solve()

    if MPI.COMM_WORLD.Get_rank() == 0:
        files = []
        path = 'results/arenstorf/'
        # Load MGRIT approximation of the solution
        for filename in os.listdir(path):
            files.append(
                [int(filename[filename.find('-') + 1: -4]), np.load(path + filename, allow_pickle=True).tolist()])
        files.sort(key=lambda tup: tup[0])
        sol = [l.pop(1) for l in files]
        sol_flat = [item for sublist in sol for item in sublist]
        # Plot MGRIT approximation of Arenstorf orbit
        plt.plot(0, 0, marker='o', color='black')
        plt.plot(1, 0, marker='o', color='black')
        plt.text(0.1, 0.1, u'Earth')
        plt.text(1.0, 0.1, u'Moon')
        # Use member function of VectorArenstorf to plot orbit
        for i in range(0, len(sol_flat), 1000):
            sol_flat[i].plot()
        plt.show()


if __name__ == '__main__':
    main()
