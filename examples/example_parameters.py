from mpi4py import MPI

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_structure,       # Problem structure
                  transfer=None,                                # Spatial grid transfer. Automatically set if None.
                  it = 10,                                      # Maximal number of iterations
                  tol=1e-10,                                    # Stopping tolerance
                  nested_iteration=True,                        # Nested iterations
                  cf_iter=1,                                    # Number of CF iterations per relaxation step
                  cycle_type='V',                               # 'V' or 'F' relaxation
                  comm_time=MPI.COMM_WORLD,                     # Time communicator
                  comm_space = MPI.COMM_NULL,                   # Space communicator
                  logging_lvl=20,                               # Logging level:
                                                                # 00 - 10: Debug -> Runtime of all components
                                                                # 11 - 20: Info  -> Info per iteration + summary
                                                                # 31 - 50: None  -> No information
                  output_fcn=None,                              # Save solutions to file
                  output_lvl=1,                                 # Output level:
                                                                # 0 -> output_fcn is never called
                                                                # 1 -> output_fcn is called when solve stops
                                                                # 2 -> output_fcn is called after each MGRIT iteration
                  random_init_guess=False                       # Random initial guess of all unknowns?
                  )

    # Solve
    return mgrit.solve()


if __name__ == '__main__':
    main()
