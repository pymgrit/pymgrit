"""
Use PyMGRIT's routines simple_setup_problem() and Mgrit() to
generate a multigrid hierarchy and MGRIT solver and run the
solver routine mgrit.solve().
"""

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the solver tolerance to 1e-10
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

    # Solve the test problem
    return mgrit.solve()

if __name__ == '__main__':
    main()
