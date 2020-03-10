import pathlib
import numpy as np
from pymgrit.lorentz.lorentz import Lorentz
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():
    # Creating the finest level problem
    lorentz_lvl_0 = Lorentz(t_start=0, t_stop=40, nt=14401)
    lorentz_lvl_1 = Lorentz(t_interval=lorentz_lvl_0.t[::80])

    # Setup the multilevel structure by using the simple_setup_problem function
    lorentz_multilevel_structure = simple_setup_problem(problem=lorentz_lvl_0, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=[lorentz_lvl_0, lorentz_lvl_1], output_lvl=2, cf_iter=0)

    # Solve
    return mgrit.solve()

if __name__ == '__main__':
    main()
