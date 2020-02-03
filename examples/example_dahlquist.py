from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():
    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)
    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_strucutre = simple_setup_problem(problem=dahlquist, level=2,coarsening=2)
    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_strucutre, tol = 1e-10)
    # Solve
    return mgrit.solve()


if __name__ == '__main__':
    main()
