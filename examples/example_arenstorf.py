"""
Apply two-level MGRIT with F-relaxation
to compute an Arenstorf orbit, using RK4 time integration
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pymgrit.arenstorf_orbit.arenstorf_orbit import ArenstorfOrbit
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'arenstorf' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file; here, we have four solution values at each time point.
        np.save(path + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),  # Local time interval
                [self.u[0][i] for i in self.index_local[0]])                # Save solution values at local time points

    # Create two-level time-grid hierarchy for the ODE system describing Arenstorf orbits
    ahrenstorf_lvl_0 = ArenstorfOrbit(t_start=0, t_stop=17.06521656015796, nt=80001)
    ahrenstorf_lvl_1 = ArenstorfOrbit(t_interval=ahrenstorf_lvl_0.t[::320])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[ahrenstorf_lvl_0, ahrenstorf_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0)

    # Compute Arenstorf orbit
    infos = mgrit.solve()

    res = np.load('results/arenstorf/' + str(len(infos['conv'])) + '/0.0:17.06521656015796.npy', allow_pickle=True)

    # Plot MGRIT approximation of Arenstorf orbit
    plt.plot(0, 0, marker='o', color='black')
    plt.plot(1, 0, marker='o', color='black')
    plt.text(0.1, 0.1, u'Earth')
    plt.text(1.0, 0.1, u'Moon')
    # Use member function of VectorArenstorf to plot orbit
    for i in range(0, len(res), 1000):
        res[i].plot()
    plt.show()
    print('a')


if __name__ == '__main__':
    main()
