"""
Solver parameters
"""

from mpi4py import MPI

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem
    mgrit = Mgrit(problem=dahlquist_multilevel_structure,       # Problem structure
                  transfer=None,                                # Spatial grid transfer. Automatically set if None.
                  it=10,                                        # Maximum number of iterations
                  tol=1e-10,                                    # Stopping tolerance
                  nested_iteration=True,                        # Use nested iterations
                  cf_iter=1,                                    # Number of FC relaxations
                  cycle_type='V',                               # multigrid cycling type:
                                                                # 'V' -> V-cycles
                                                                # 'F' -> F-cycles
                  comm_time=MPI.COMM_WORLD,                     # Time communicator
                  comm_space=MPI.COMM_NULL,                     # Space communicator
                  logging_lvl=20,                               # Logging level:
                                                                # 00 - 10: Debug -> Runtime of all components
                                                                # 11 - 20: Info  -> Info per iteration + summary
                                                                # 31 - 50: None  -> No information
                  output_fcn=None,                              # Save solutions to file
                  output_lvl=1,                                 # Output level:
                                                                # 0 -> output_fcn is never called
                                                                # 1 -> output_fcn is called at the end of the simulation
                                                                # 2 -> output_fcn is called after each MGRIT iteration
                  random_init_guess=False                       # Use random initial guess for all unknowns
                  )

    # Solve the test problem
    return mgrit.solve()


if __name__ == '__main__':
    main()
