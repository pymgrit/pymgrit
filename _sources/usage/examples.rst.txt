**********
Examples
**********

This page contains short examples that demonstrate basic and advanced usage of PyMGRIT.
The source code for these and more examples is available in the examples_ folder.

.. _examples: https://github.com/pymgrit/pymgrit/tree/master/examples

-----------------
Table of Contents
-----------------

    - `Basic usage`_
    - `Multigrid hierarchy`_
    - `Solver parameters`_
    - `Output function`_
    - `Spatial transfer operator`_


-----------
Basic usage
-----------

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

Note: Option 1 is only implemented to support an easy start. We recommend to build the hierarchy manually by using one
of the options 2-4.

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

-------------------------
Spatial transfer operator
-------------------------

example_spatial_coarsening.py_

.. _example_spatial_coarsening.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_spatial_coarsening.py

This example shows how the transfer parameter of the MGRIT solver can be used to perform an additional spatial
coarsening on the different levels. We use the 1D heat equation (see :doc:`../applications/heat_equation`).

The first step is to import all necessary PyMGRIT classes::

    import numpy as np

    from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
    from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
    from pymgrit.core.mgrit import Mgrit  # MGRIT solver
    from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
    from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class

Then, we define the class GridTransferHeat for the 1D heat equation::

    class GridTransferHeat(GridTransfer):
        """
        Grid Transfer for the Heat Equation.
        Interpolation: Linear interpolation
        Restriction: Full weighted
        """

        def __init__(self):
            """
            Constructor.
            :rtype: object
            """
            super(GridTransferHeat, self).__init__()

The grid transfer class must contain two member functions: `restriction` and `interpolation`.

The function restriction receives a VectorHeat1D object and returns another VectorHeat1D object, that contains
the restricted solution vector::

    def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Restrict u using full weighting.

        Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
              The Heat1D vector class stores only the non boundary points.
        :param u: VectorHeat1D
        :rtype: VectorHeat1D
        """
        # Get the non boundary points
        sol = u.get_values()

        # Create array
        ret_array = np.zeros(int((len(sol) - 1) / 2))

        # Full weighting
        for i in range(len(ret_array)):
            ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

        # Create and return a VectorHeat1D object
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

The function interpolation works in the same way::

    def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Interpolate u using linear interpolation

        Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
              The Heat1D vector class stores only the non boundary points.
        :param u: VectorHeat1D
        :rtype: VectorHeat1D
        """
        # Get the non boundary points
        sol = u.get_values()

        # Create array
        ret_array = np.zeros(int(len(sol) * 2 + 1))

        # Linear interpolation
        for i in range(len(sol)):
            ret_array[i * 2] += 1 / 2 * sol[i]
            ret_array[i * 2 + 1] += sol[i]
            ret_array[i * 2 + 2] += 1 / 2 * sol[i]

        # Create and return a VectorHeat1D object
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

Now, we construct our multilevel scheme building each level. In this example, we use four-level MGRIT. The finest level
has 17 points in space, the second level 9, the third level 5 and the fourth level also 5.

Note: In this example, it is not possible to use the PyMGRIT's core function simple_setup_problem, since each level
has different spatial sizes::

    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

Then, we have to define the transfer operator per grid level. The transfer operator is a list of lengths (#level -1) and
specifies the transfer operator used per level. For the transfer between the first and the second level,
an object of the class GridTransferHeat() is needed to transfer the solution between the different space grids with
different sizes. The same is necessary for the transfer between the second and third level. Since the third and fourth
level have the same size in space, the GridTransferCopy class from PyMGRIT's core is used. Set up the MGRIT solver with
the problem and the transfer operators and solve the problem::

    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]
    mgrit = Mgrit(problem=problem, transfer=transfer)
    info = mgrit.solve()


Complete code::

    import numpy as np

    from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
    from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
    from pymgrit.core.mgrit import Mgrit  # MGRIT solver
    from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
    from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class

    class GridTransferHeat(GridTransfer):
        """
        Grid Transfer for the Heat Equation.
        Interpolation: Linear interpolation
        Restriction: Full weighted
        """

        def __init__(self):
            """
            Constructor.
            :rtype: object
            """
            super(GridTransferHeat, self).__init__()

        # Specify restriction operator
        def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
            """
            Restrict u using full weighting.

            Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
                  The Heat1D vector class stores only the non boundary points.
            :param u: VectorHeat1D
            :rtype: VectorHeat1D
            """
            # Get the non boundary points
            sol = u.get_values()

            # Create array
            ret_array = np.zeros(int((len(sol) - 1) / 2))

            # Full weighting
            for i in range(len(ret_array)):
                ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

            # Create and return a VectorHeat1D object
            ret = VectorHeat1D(len(ret_array))
            ret.set_values(ret_array)
            return ret

        # Specify interpolation operator
        def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
            """
            Interpolate u using linear interpolation

            Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
                  The Heat1D vector class stores only the non boundary points.
            :param u: VectorHeat1D
            :rtype: VectorHeat1D
            """
            # Get the non boundary points
            sol = u.get_values()

            # Create array
            ret_array = np.zeros(int(len(sol) * 2 + 1))

            # Linear interpolation
            for i in range(len(sol)):
                ret_array[i * 2] += 1 / 2 * sol[i]
                ret_array[i * 2 + 1] += sol[i]
                ret_array[i * 2 + 2] += 1 / 2 * sol[i]

            # Create and return a VectorHeat1D object
            ret = VectorHeat1D(len(ret_array))
            ret.set_values(ret_array)
            return ret

    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

    # Specify a list of grid transfer operators of length (#level - 1)
    # Using the new class GridTransferHeat to apply spatial coarsening on the first two levels
    # Using the PyMGRIT's core class GridTransferCopy on the last level (no spatial coarsening)
    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]

    # Setup MGRIT solver with problem and transfer
    mgrit = Mgrit(problem=problem, transfer=transfer)

    info = mgrit.solve()