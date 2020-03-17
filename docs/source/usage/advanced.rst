**************
Advanced usage
**************

This page contains short examples that demonstrate basic advanced of PyMGRIT.
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


PyMGRIT allows different application classes and/or time integration schemes in the multigrid hierarchy. The first_
examples shows how to implement different time integrator methods in an application class. The second_
examples shows how to use multiple application classes in the multigrid hierarchy.

.. _first: https://github.com/pymgrit/pymgrit/tree/master/examples/example_time_integrators.py
.. _second: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_1d_bdf2.py

The step function can contain multiple integration methods, that can be choosen by a parameter. The `Dahlquist
application class`_ implements the following time integration routines:

.. _Dahlquist  application class: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/dahlquist/dahlquist.py

* Backward Euler
* Forward Euler
* Trapezoidal rule
* implicit Mid-point rule

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

The corresponding example_ creates a two-level hierarchy in the usual way, using the Mid-point rule at the first level
and backward Euler at the second level.

.. _example: https://github.com/pymgrit/pymgrit/tree/master/examples/example_time_integrators.py

::

    # Create Dahlquist's test problem choosing implicit mid-point rule as time stepper
    dahlquist_lvl0 = Dahlquist(t_start=0, t_stop=5, nt=101, method='MR')
    # Create Dahlquist's test problem choosing implicit backward euler as time stepper
    dahlquist_lvl1 = Dahlquist(t_start=0, t_stop=5, nt=51, method='BE')

    # Setup MGRIT and solve the problem
    mgrit = Mgrit(problem=[dahlquist_lvl0, dahlquist_lvl1])
    info = mgrit.solve()

In the second example, we use two application classes for generating different time integration routines on different
levels. The first application class implements BDF2_ for the 1D heat equation example, while the second class
implements BDF1_. Both application classes share the same Vector structure for the solution of two time-points in one
vector. The corresponding file_ builds a hierarchy of the levels with the BDF2 application class on the first level
and the BDF1 application class on the second and third level.

.. _BDF2: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/heat/heat_1d_2pts_bdf2.py
.. _BDF1: https://github.com/pymgrit/pymgrit/blob/master/src/pymgrit/heat/heat_1d_2pts_bdf1.py
.. _file: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_1d_bdf2.py

::

    def rhs(x, t):
        """
        Right-hand side of 1D heat equation example problem at a given space-time point (x,t)
        :param x: spatial grid point
        :param t: time point
        :return: right-hand side of 1D heat equation example problem at point (x,t)
        """

        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def init_con_fnc(x, t):
        """
        Exact solution of 1D heat equation example problem at a given space-time point (x,t)
        :param x: spatial grid point
        :param t: time point
        :return: exact solution of 1D heat equation example problem at point (x,t)
        """
        return np.sin(np.pi * x) * np.cos(t)

    t_stop = 2
    nt = 512
    dt = t_stop / nt
    t_interval = np.linspace(0, t_stop, int(nt / 2 + 1))
    heat0 = Heat1DBDF2(x_start=0, x_end=1, nx=1001, a=1, dt=dt, rhs=rhs, init_con_fnc=init_con_fnc,
                       t_interval=t_interval)
    heat1 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dt=dt, rhs=rhs, init_con_fnc=init_con_fnc,
                       t_interval=heat0.t[::2])
    heat2 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dt=dt, rhs=rhs, init_con_fnc=init_con_fnc,
                       t_interval=heat1.t[::2])

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    mgrit.solve()


------------------
Spatial coarsening
------------------

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

