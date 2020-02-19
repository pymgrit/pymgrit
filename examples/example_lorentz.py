import pathlib
import numpy as np
from pymgrit.lorentz.lorentz import Lorentz
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        #Set path to solution
        path = 'results/' + 'lorentz' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution to file.
        np.save(path + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),  # Add time information for distinguish procs
                [self.u[0][i].get_values() for i in self.index_local[0]])

    # Creating the finest level problem
    lorentz_lvl_0 = Lorentz(t_start=0, t_stop=40, nt=14401)
    lorentz_lvl_1 = Lorentz(t_interval=lorentz_lvl_0.t[::80])

    # Setup the multilevel structure by using the simple_setup_problem function
    lorentz_multilevel_structure = simple_setup_problem(problem=lorentz_lvl_0, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=[lorentz_lvl_0, lorentz_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0)

    # Solve
    return mgrit.solve()

if __name__ == '__main__':
    main()
