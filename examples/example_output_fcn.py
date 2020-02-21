import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():
    # Define output function that writes the solution to a file
    def output_fcn(self):
        #Set path to solution
        path = 'results/' + 'dahlquist'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file.
        np.save(path + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),  # Add time information for distinguish procs
                [self.u[0][i].get_values() for i in self.index_local[0]])   # Save each time step per processors

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, output_fcn=output_fcn)

    # Solve
    info = mgrit.solve()

    # Plot solution if one processor was used
    res = np.load('results/dahlquist/0.0:5.0.npy')
    plt.plot(res)
    plt.show()


if __name__ == '__main__':
    main()
