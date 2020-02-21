**********
Examples
**********

Short usage examples for PyMGRIT. The source code for these and more examples can is available in the examples_ folder.

.. _examples: https://github.com/pymgrit/pymgrit/tree/master/examples

-----------------
Table of Contents
-----------------

    - `Simple solve`_
    - `MGRIT Parameters`_
    - `Output function`_
    - `Define multigrid structure`_
    - `Spatial transfer operator`_
    - `Space & time parallelism`_


------------
Simple solve
------------

::

    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

    # Solve
    mgrit.solve()

----------------
MGRIT Parameters
----------------

The MGRIT algorithm in PyMGRIT has a variety of parameters and features. The example 'example_parameters' describes these and gives an overview of what is possible with PyMGRIT:

::

    from mpi4py import MPI

    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_structure,   # Problem structure
                  transfer=None,                            # Spatial grid transfer. Automatically set if None.
                  it=10,                                    # Maximal number of iterations
                  tol=1e-10,                                # Stopping tolerance
                  nested_iteration=True,                    # Nested iterations
                  cf_iter=1,                                # Number of CF iterations per relaxation step
                  cycle_type='V',                           # 'V' or 'F' relaxation
                  comm_time=MPI.COMM_WORLD,                 # Time communicator
                  comm_space = MPI.COMM_NULL,               # Space communicator
                  logging_lvl=50,                           # Logging level:
                                                            # 00 - 10: Debug -> Runtime of all components
                                                            # 11 - 20: Info  -> Info per iteration + summary
                                                            # 31 - 50: None  -> No information
                  output_fcn=None,                          # Save solutions to file
                  output_lvl=1,                             # Output level:
                                                            # 0 -> output_fcn is never called
                                                            # 1 -> output_fcn is called when solve stops
                                                            # 2 -> output_fcn is called after each MGRIT iteration
                  random_init_guess=False                   # Random initial guess of all unknowns?
                  )

    # Solve
    mgrit.solve()

---------------
Output function
---------------

To store the solutions an output function must be written, which is passed to the MGRIT algorithm. The output function is called in the algorithm after each iteration, at the end or not at all, depending on the setting (see example parameter). The output function is called on each processor. In the example the solution is written to a file via the numpy function save.

::

    import pathlib
    import matplotlib.pyplot as plt
    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

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

--------------------------
Define multigrid structure
--------------------------

There are several ways to create a multi-level structure that can be solved by the MGRIT algorithm:

- Using the simple setup function
- Setup each level by t_start, t_end, nt
- Setup by intervals
- Mix

The following example shows the different possibilities:

::

    import numpy as np
    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

    # Different ways for creating the multilevel structure

    # Variant 1: Simple setup
    dahlquist_multilevel_structure_1 = simple_setup_problem(problem=Dahlquist(t_start=0, t_stop=5, nt=101), level=3,
                                                            coarsening=2)
    Mgrit(problem=dahlquist_multilevel_structure_1, tol=1e-10).solve()

    # Variant 2: Setup each level by t_start, t_end, nt
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_start=0, t_stop=5, nt=51)
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_2 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_2, tol=1e-10).solve()

    # Variant 3: Setup by intervals
    t_interval = np.linspace(0, 5, 101)
    dahlquist_lvl_0 = Dahlquist(t_interval=t_interval)
    dahlquist_lvl_1 = Dahlquist(t_interval=t_interval[::2])  # Takes every second point from t_interval
    dahlquist_lvl_2 = Dahlquist(t_interval=t_interval[::4])  # Takes every fourth point from t_interval
    dahlquist_multilevel_structure_3 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_3, tol=1e-10).solve()

    # Variant 4: Mix
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_interval=dahlquist_lvl_0.t[::2])  # Using t from the upper level.
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_4 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_4, tol=1e-10).solve()

-------------------------
Spatial transfer operator
-------------------------

TODO

------------------------
Space & time parallelism
------------------------

TODO
