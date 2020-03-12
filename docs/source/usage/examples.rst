**********
Examples
**********

This page contains short examples that demonstrate basic and advanced usage of PyMGRIT.
The source code for these and more examples is available in the examples_ folder.

.. _examples: https://github.com/pymgrit/pymgrit/tree/master/examples

-----------------
Table of Contents
-----------------

    - `Simple solve and PyMGRIT output`_
    - `Solver parameters`_
    - `Output function`_
    - `Multigrid hierarchy`_
    - `Spatial transfer operator`_


-------------------------------
Simple solve and PyMGRIT output
-------------------------------

example_dahlquist.py_

.. _example_dahlquist.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_dahlquist.py

This example demonstrates basic usage of the PyMGRIT package for solving a simple test problem with a two-level MGRIT solver,
using PyMGRIT's core routines `simple_setup_problem()`, `Mgrit()`, and `mgrit.solve()`.
Note: This example is also considered in :doc:`Quickstart <quickstart>`.

For a given test problem, we can construct a time-multigrid hierarchy by calling `simple_setup_problem()`.
To use `mgrit.solve()` we then only need to set up an MGRIT solver with this time-multigrid hierarchy.

::

    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the solver tolerance to 1e-10
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

    # Solve the test problem
    info = mgrit.solve()

produces the output::

    INFO - 03-02-20 11:19:03 - Start setup
    INFO - 03-02-20 11:19:03 - Setup took 0.009920358657836914 s
    INFO - 03-02-20 11:19:03 - Start solve
    INFO - 03-02-20 11:19:03 - iter 1  | conv: 7.186185937031941e-05  | conv factor: -                       | runtime: 0.01379704475402832 s
    INFO - 03-02-20 11:19:03 - iter 2  | conv: 1.2461067076355103e-06 | conv factor: 0.017340307063501627    | runtime: 0.007235527038574219 s
    INFO - 03-02-20 11:19:03 - iter 3  | conv: 2.1015566145245807e-08 | conv factor: 0.016864981158092696    | runtime: 0.005523681640625 s
    INFO - 03-02-20 11:19:03 - iter 4  | conv: 3.144127445017594e-10  | conv factor: 0.014960945726074891    | runtime: 0.004599332809448242 s
    INFO - 03-02-20 11:19:03 - iter 5  | conv: 3.975214076032893e-12  | conv factor: 0.01264329816633959     | runtime: 0.0043201446533203125 s
    INFO - 03-02-20 11:19:03 - Solve took 0.042092084884643555 s
    INFO - 03-02-20 11:19:03 - Run parameter overview

    time interval             : [0.0, 5.0]
    number of time points     : 101
    max dt                    : 0.05000000000000071
    number of levels          : 2
    coarsening factors        : [2]
    cf_iter                   : 1
    nested iteration          : True
    cycle type                : V
    stopping tolerance        : 1e-10
    time communicator size    : 1
    space communicator size   : -99

and returns the residual history, setup time, and solve time in dictionary `info` with the following key values:

    - `conv` : residual history (2-norm of the residual at each iteration)
    - `time_setup` : setup time [in seconds]
    - `time_solve` : solve time [in seconds]

-----------------
Solver parameters
-----------------

example_parameters.py_

.. _example_parameters.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_parameters.py

The MGRIT algorithm in PyMGRIT has a variety of parameters and features. This example describes the parameters
of PyMGRIT's core routine `Mgrit()`.

::

    from mpi4py import MPI

    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

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
                  random_init_guess=False                       # Use (True) or do not use (False) random initial guess
                                                                # for all unknowns (default: False)
                  )

    # Solve the test problem
    mgrit.solve()


---------------
Output function
---------------

example_output_fcn_serial.py_ and example_output_fcn.py_

.. _example_output_fcn_serial.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_output_fcn_serial.py
.. _example_output_fcn.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_output_fcn.py

In this example, we show how to save and plot the MGRIT approximation of the solution of Dahlquist's test problem.
An output function is defined that saves the solution (here, a single solution value at each time point is written to an
output file via the ``numpy`` function `save()`). This output function is passed to the MGRIT solver.
Depending on the solver setting (see `output_lvl` in `Solver parameters`_), the output function

* is never called,

* is called at the end of the simulation (example_output_fcn_serial.py_), or

* is called after each iteration (example_output_fcn.py_).


::

    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt

    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit


    # Define output function that writes the solution to a file
    def output_fcn(self):
        # Set path to solution
        path = 'results/' + 'dahlquist'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Save solution to file; here, we just have a single solution value at each time point.
        # Useful member variables of MGRIT solver:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        np.save(path + '/dahlquist',
                [self.u[0][i].get_values() for i in self.index_local[0]])   # Solution values at local time points

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the output function
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, output_fcn=output_fcn)

    # Solve the test problem
    info = mgrit.solve()

    # Plot the solution (Note: modifications necessary if more than one process is used for the simulation!)
    t = np.linspace(dahlquist.t_start, dahlquist.t_end, dahlquist.nt)
    sol = np.load('results/dahlquist/dahlquist.npy')
    plt.plot(t, sol)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.show()

-------------------
Multigrid hierarchy
-------------------

example_multilevel_structure.py_

.. _example_multilevel_structure.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_multilevel_structure.py

There are several ways to create a time-multigrid hierarchy for a problem:

#. Using PyMGRIT's core function `simple_setup_problem()`
#. Defining `nt` evenly spaced numbers over a specified interval `[t_start, t_stop]` for each grid level
#. Specifying time intervals for each grid level
#. Mixing options 2 and 3

The following example shows the four different options and builds MGRIT solvers using the resulting four multilevel objects:

::

    import numpy as np
    from pymgrit.dahlquist.dahlquist import Dahlquist
    from pymgrit.core.simple_setup_problem import simple_setup_problem
    from pymgrit.core.mgrit import Mgrit

    # Option 1: Use PyMGRIT's core function simple_setup_problem()
    dahlquist_multilevel_structure_1 = simple_setup_problem(problem=Dahlquist(t_start=0, t_stop=5, nt=101), level=3,
                                                            coarsening=2)
    Mgrit(problem=dahlquist_multilevel_structure_1, tol=1e-10).solve()

    # Option 2: Build each level using t_start, t_end, and nt
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_start=0, t_stop=5, nt=51)
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_2 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_2, tol=1e-10).solve()

    # Option 3: Specify time intervals for each grid level
    t_interval = np.linspace(0, 5, 101)
    dahlquist_lvl_0 = Dahlquist(t_interval=t_interval)
    dahlquist_lvl_1 = Dahlquist(t_interval=t_interval[::2])  # Takes every second point from t_interval
    dahlquist_lvl_2 = Dahlquist(t_interval=t_interval[::4])  # Takes every fourth point from t_interval
    dahlquist_multilevel_structure_3 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_3, tol=1e-10).solve()

    # Option 4: Mix options 2 and 3
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_interval=dahlquist_lvl_0.t[::2])  # Using t from the upper level.
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_4 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_4, tol=1e-10).solve()

-------------------------
Spatial transfer operator
-------------------------

TODO

