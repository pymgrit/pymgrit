"""
MGRIT solver parameters
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
                  max_iter=10,                                  # Maximum number of iterations (default: 100)
                  tol=1e-10,                                    # Stopping tolerance (default: 1e-7)
                  nested_iteration=True,                        # Use (True) or do not use (False) nested iterations
                                                                # (default: True)
                  cf_iter=1,                                    # Number of CF relaxations (default: 1)
                  cycle_type='V',                               # multigrid cycling type (default: 'V'):
                                                                # 'V' -> V-cycles
                                                                # 'F' -> F-cycles
                  comm_time=MPI.COMM_WORLD,                     # Time communicator (default: MPI.COMM_WORLD)
                  comm_space=MPI.COMM_NULL,                     # Space communicator (default: MPI.COMM_NULL)
                  weight_c=1,                                   # C - relaxation weight (default: 1)
                  logging_lvl=20,                               # Logging level (default: 20):
                                                                # 10: Debug -> Runtime of all components
                                                                # 20: Info  -> Info per iteration + summary
                                                                # 30: None  -> No information
                  output_fcn=None,                              # Function for saving solution values to file
                                                                # (default: None)
                  output_lvl=1,                                 # Output level (default: 1):
                                                                # 0 -> output_fcn is never called
                                                                # 1 -> output_fcn is called at the end of the simulation
                                                                # 2 -> output_fcn is called after each MGRIT iteration
                  t_norm=2,                                     # Temporal norm
                                                                # 1 -> One-norm
                                                                # 2 -> Two-norm
                                                                # 3 -> Infinity-norm
                  random_init_guess=False                       # Use (True) or do not use (False) random initial guess
                                                                # for all unknowns (default: False)
                  )

    # Solve the test problem
    return mgrit.solve()


if __name__ == '__main__':
    main()
