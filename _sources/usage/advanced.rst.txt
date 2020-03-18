**************
Advanced usage
**************

This page contains short examples that demonstrate advanced usage of PyMGRIT.
The source code for these and more examples is available in the examples_ folder.

.. _examples: https://github.com/pymgrit/pymgrit/tree/master/examples

    - `Advanced multigrid hierarchy`_
    - `Spatial coarsening`_
    - `Convergence criteria`_


----------------------------
Advanced multigrid hierarchy
----------------------------

example_time_integrators.py_ and example_heat_1d_bdf2.py_

.. _example_time_integrators.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_time_integrators.py
.. _example_heat_1d_bdf2.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_1d_bdf2.py


PyMGRIT allows using different application classes and/or time integration schemes in the multigrid hierarchy.

* Example 1 shows how to implement different time integration methods in an application class.
* Example 2 shows how to use multiple application classes in the multigrid hierarchy.

**Example 1** One application class with different time integration methods

The member function `step()` of an application class can carry out a time integration step based on different time
integration methods. The `Dahlquist
application class`_ implements the following time integration schemes:

.. _Dahlquist  application class: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/dahlquist/dahlquist.py

* Backward Euler
* Forward Euler
* Trapezoidal rule
* Implicit mid-point rule

that can be controlled by the member variable `method`:

::

    def step(self, u_start: VectorDahlquist, t_start: float, t_stop: float) -> VectorDahlquist:
    """
    Time integration routine for Dahlquist's test problem:
        BE: Backward Euler
        FE: Forward Euler
        TR: Trapezoidal rule
        MR: implicit Mid-point rule

    :param u_start: approximate solution for the input time t_start
    :param t_start: time associated with the input approximate solution u_start
    :param t_stop: time to evolve the input approximate solution to
    :return: approximate solution for the input time t_stop
    """
    z = (t_stop - t_start) * -1  # Note: lambda = -1
    if self.method == 'BE':
        tmp = 1 / (1 - z) * u_start.get_values()
    elif self.method == 'FE':
        tmp = (1 + z) * u_start.get_values()
    elif self.method == 'TR':
        tmp = (1 + z / 2) / (1 - z / 2) * u_start.get_values()
    elif self.method == 'MR':
        k1 = -1 / (1 - z / 2) * u_start.get_values()
        tmp = u_start.get_values() + (t_stop - t_start) * k1
    return VectorDahlquist(tmp)

The corresponding example (example_time_integrators.py_) creates a two-level hierarchy for Dahlquist's test problem, using the implicit mid-point
rule on the fine grid (level 0) and backward Euler on the coarse grid (level 1):

::

    # Create Dahlquist's test problem using implicit mid-point rule time integration
    dahlquist_lvl0 = Dahlquist(t_start=0, t_stop=5, nt=101, method='MR')
    # Create Dahlquist's test problem using backward Euler time integration
    dahlquist_lvl1 = Dahlquist(t_start=0, t_stop=5, nt=51, method='BE')

    # Setup an MGRIT solver and solve the problem
    mgrit = Mgrit(problem=[dahlquist_lvl0, dahlquist_lvl1])
    info = mgrit.solve()

**Example 2** Two application classes

In the second example, we use two application classes for implementing two different
time integration methods for the 1D heat equation example:

* Application class 1 implements BDF2_.
* Application class 2 implements BDF1_.

Note: The `vector class`_ used in both application classes contains the solution at two consecutive time points.

.. _BDF2: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/heat/heat_1d_2pts_bdf2.py
.. _BDF1: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/heat/heat_1d_2pts_bdf1.py
.. _`vector class`: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/heat/vector_heat_1d_2pts.py

The corresponding example (example_heat_1d_bdf2.py_) constructs a three-level multigrid hierarchy for the 1D heat
equation example using the BDF2 application class on the fine grid (level 0) and the BDF1 application class on the
first and second coarse grids (levels 1 and 2):

::

    def rhs(x, t):
        """
        Right-hand side of 1D heat equation example problem at a given space-time point (x,t),
          -sin(pi*x)(sin(t) - a*pi^2*cos(t)),  a = 1

        :param x: spatial grid point
        :param t: time point
        :return: right-hand side of 1D heat equation example problem at point (x,t)
        """

        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def init_cond(x):
        """
        Initial condition of 1D heat equation example,
          u(x,0)  = sin(pi*x)

        :param x: spatial grid point
        :return: initial condition of 1D heat equation example problem
        """
        return np.sin(np.pi * x)

    # Time interval
    t_start = 0
    t_stop = 2
    nt = 512  # number of time points excluding t_start
    dt = t_stop / nt  # time-step size

    # Time points are grouped in pairs of two consecutive time points
    #   => (nt/2) + 1 pairs
    # Note: * Each pair is associated with the time value of its first point.
    #       * The second value of the last pair (associated with t_stop) is not used.
    #       * The spacing within each pair is the same (= dt) on all grid levels.
    t_interval = np.linspace(t_start, t_stop, int(nt / 2 + 1))

    heat0 = Heat1DBDF2(x_start=0, x_end=1, nx=1001, a=1, dtau=dt, rhs=rhs, init_cond=init_cond,
                       t_interval=t_interval)
    heat1 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dtau=dt, rhs=rhs, init_cond=init_cond,
                       t_interval=heat0.t[::2])
    heat2 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dtau=dt, rhs=rhs, init_cond=init_cond,
                       t_interval=heat1.t[::2])

    # Setup three-level MGRIT solver and solve the problem
    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    info = mgrit.solve()


------------------
Spatial coarsening
------------------

example_spatial_coarsening.py_

.. _example_spatial_coarsening.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_spatial_coarsening.py

This example demonstrates how to use the transfer parameter `transfer` of the MGRIT solver to apply spatial
coarsening on different levels of the time-grid hierarchy for solving a 1D heat equation problem (see :doc:`../applications/heat_equation`).

The first step is to import all necessary PyMGRIT classes (and ``numpy`` for later use)::

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
        Restriction: Full weighting
        """

        def __init__(self):
            """
            Constructor.
            :rtype: GridTransferHeat object
            """
            super(GridTransferHeat, self).__init__()

The grid transfer class must contain the two member functions `restriction()` and `interpolation()`.

The function `restriction()` receives a `VectorHeat1D` object and returns another `VectorHeat1D` object that contains
the restricted solution vector::

    def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Restrict input vector u using standard full weighting restriction.

        Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
              The Heat1D vector class only stores interior points.
        :param u: approximate solution vector
        :return: input solution vector u restricted to a coarse grid
        """
        # Get values at interior points
        sol = u.get_values()

        # Create array for restricted values
        ret_array = np.zeros(int((len(sol) - 1) / 2))

        # Full weighting restriction
        for i in range(len(ret_array)):
            ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

        # Create and return a VectorHeat1D object with the restricted values
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

Similarly, we define the function `interpolation()` as follows::

    def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Interpolate input vector u using linear interpolation.

        Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
              The Heat1D vector class only stores interior points.
        :param u: approximate solution vector
        :return: input solution vector u interpolated to a fine grid
        """
        # Get values at interior points
        sol = u.get_values()

        # Create array for interpolated values
        ret_array = np.zeros(int(len(sol) * 2 + 1))

        # Linear interpolation
        for i in range(len(sol)):
            ret_array[i * 2] += 1 / 2 * sol[i]
            ret_array[i * 2 + 1] += sol[i]
            ret_array[i * 2 + 2] += 1 / 2 * sol[i]

        # Create and return a VectorHeat1D object with interpolated values
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

Now, we construct a multigrid hierarchy for the 1d heat example. Here, we set up the following hierarchy:

  * level 0: 129 time points, 17 points in space
  * level 1: 65 time points, 9 points in space
  * level 2: 33 time points, 5 points in space
  * level 3: 17 time points, 5 points in space

Note: In this example, it is not possible to use PyMGRIT's core function `simple_setup_problem()`, since the number of
spatial grid points changes in the multigrid hiearchy::

    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

Before we can set up the MGRIT solver, we have to define the grid transfer between all two consecutive levels in the
multigrid hierarchy. These grid transfers are specified by a list of grid transfer objects of length (#levels -1).
For our four-level example, this list is of length three with two objects of the new class `GridTransferHeat` for the
transfer between levels 0 and 1 as well as between levels 1 and 2 and an object of PyMGRIT's core class
`GridTransferCopy` for the transfer between levels 2 and 3::

    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]

Finally, we set up the MGRIT solver and solve the problem::

    mgrit = Mgrit(problem=problem, transfer=transfer)
    info = mgrit.solve()


Complete code::

    import numpy as np

    from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
    from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
    from pymgrit.core.mgrit import Mgrit  # MGRIT solver
    from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
    from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class


    # Create class for the grid transfer between spatial grids.
    # Note: The class must inherit from PyMGRIT's core GridTransfer class.
    class GridTransferHeat(GridTransfer):
        """
        Grid Transfer class for the Heat Equation.
        Interpolation: Linear interpolation
        Restriction: Full weighting
        """

        def __init__(self):
            """
            Constructor.
            :rtype: GridTransferHeat object
            """
            super(GridTransferHeat, self).__init__()

        # Define restriction operator
        def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
            """
            Restrict input vector u using standard full weighting restriction.

            Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
                  The Heat1D vector class only stores interior points.
            :param u: approximate solution vector
            :return: input solution vector u restricted to a coarse grid
            """
            # Get values at interior points
            sol = u.get_values()

            # Create array for restricted values
            ret_array = np.zeros(int((len(sol) - 1) / 2))

            # Full weighting restriction
            for i in range(len(ret_array)):
                ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

            # Create and return a VectorHeat1D object with the restricted values
            ret = VectorHeat1D(len(ret_array))
            ret.set_values(ret_array)
            return ret

        # Define interpolation operator
        def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
            """
            Interpolate input vector u using linear interpolation.

            Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
                  The Heat1D vector class only stores interior points.
            :param u: approximate solution vector
            :return: input solution vector u interpolated to a fine grid
            """
            # Get values at interior points
            sol = u.get_values()

            # Create array for interpolated values
            ret_array = np.zeros(int(len(sol) * 2 + 1))

            # Linear interpolation
            for i in range(len(sol)):
                ret_array[i * 2] += 1 / 2 * sol[i]
                ret_array[i * 2 + 1] += sol[i]
                ret_array[i * 2 + 2] += 1 / 2 * sol[i]

            # Create and return a VectorHeat1D object with interpolated values
            ret = VectorHeat1D(len(ret_array))
            ret.set_values(ret_array)
            return ret


    # Construct a four-level multigrid hierarchy for the 1d heat example
    #   * use a coarsening factor of 2 in time on all levels
    #   * apply spatial coarsening by a factor of 2 on the first two levels
    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

    # Specify a list of grid transfer operators of length (#levels - 1) for the transfer between two consecutive levels
    #   * Use the new class GridTransferHeat to apply spatial coarsening for transfers between the first three levels
    #   * Use PyMGRIT's core class GridTransferCopy for the transfer between the last two levels (no spatial coarsening)
    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]

    # Setup four-level MGRIT solver and solve the problem
    mgrit = Mgrit(problem=problem, transfer=transfer)

    info = mgrit.solve()

--------------------
Convergence criteria
--------------------

example_convergence_criteria.py_

.. _example_convergence_criteria.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_example_convergence_criteria.py

In this example, we use a customized version of PyMGRIT's MGRIT algorithm to change the convergence criteria. The first
step is to create a new class that inherits from the MGRIT class. Afterwards, we overwrite the convergence criteria
function that is called by the algorithm after each iteration. The new convergence criteria is the maximum norm of the
relative difference of two successive iterates at all C-points::

    class MgritCustomized(Mgrit):
        """
        Customized MGRIT with maximum norm of the relative
        difference of two successive iterates as convergence criteria
        """

        def __init__(self, *args, **kwargs) -> None:
            """
            Cumstomized MGRIT constructor
            :param args:
            :param kwargs:
            """
            # Call parent constructor
            super(MgritCustomized, self).__init__(*args, **kwargs)
            # New member variable for saving the C-points values of the last iteration
            self.last_it = []
            # Initialize the new member variable
            self.convergence_criteria(iteration=0)

        def convergence_criteria(self, iteration: int) -> None:
            """
            Stops if the maximum norm of the relative
            difference of two successive iterates
            at C-points is below the stopping tolerance.
            :param iteration: Iteration number
            """

            # Create structure on the first function call
            if len(self.last_it) != len(self.index_local_c[0]):
                self.last_it = np.zeros((len(self.index_local_c[0]), len(self.u[0][0].get_values())))
            new = np.zeros_like(self.last_it)
            j = 0
            tmp = 0
            # If process has a C-point
            if self.index_local_c[0].size > 0:
                # Loop over all C-points of the process
                for i in np.nditer(self.index_local_c[0]):
                    new[j] = self.u[0][i].get_values()
                    j = j + 1
                # Compute relative difference between two iterates
                tmp = 100 * np.max(
                    np.abs(np.abs(np.divide((new - self.last_it), new, out=np.zeros_like(self.last_it), where=new != 0))))

            # Communicate the local value
            tmp = self.comm_time.allgather(tmp)
            # Maximum norm
            self.conv[iteration] = np.max(np.abs(tmp))
            self.last_it = np.copy(new)

At last, we can use the new MGRITCustomized class to solve our problem in the usual way::

    # Create two-level time-grid hierarchy for the ODE system describing Arenstorf orbits
    ahrenstorf_lvl_0 = ArenstorfOrbit(t_start=0, t_stop=17.06521656015796, nt=10001)
    ahrenstorf_lvl_1 = ArenstorfOrbit(t_interval=ahrenstorf_lvl_0.t[::100])

    # Use the customized MGRIT algorithm to solve the problem.
    # Stopps if the maximum relative change in all four variables of arenstorf orbit is smaller than 1% for all C-points
    info = MgritCustomized(problem=[ahrenstorf_lvl_0, ahrenstorf_lvl_1], tol=1).solve()

