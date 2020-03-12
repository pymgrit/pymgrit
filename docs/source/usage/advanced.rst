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