*********
Firedrake
*********

Firedrake_ is an automated system for the solution of partial differential equations using the finite element method
(FEM). In combination with PyMGRIT, Firedrake can set up and solve the spatial problem (in parallel), while PyMGRIT
takes care of the parallelization in the time dimension.

.. _Firedrake: https://www.firedrakeproject.org/index.html

------------
Installation
------------

Please follow the following steps:

* `Download and install`_ Firedrake
* Activate the virtual environment in a shell:

    >>> source firedrake/bin/activate

* Install PyMGRIT in the virtual environment

    >>> pip3 install pymgrit

That's it, you can now enjoy the benefits of both tools. Remember to activate the virtual environment in the shell
in which you want to use the coupling of both tools.

.. _Download and install: https://www.firedrakeproject.org/download.html

-----------------
Diffusion example
-----------------

* `diffusion_2d_firedrake.py`_
* `example_diffusion_2d_firedrake.py`_

.. _diffusion_2d_firedrake.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/firedrake/diffusion_2d_firedrake.py
.. _example_diffusion_2d_firedrake.py: https://github.com/pymgrit/pymgrit/blob/master/examples/firedrake/example_diffusion_2d_firedrake.py

The following example shows how to set up and solve a 2D diffusion problem using PyMGRIT and Firedrake. The spatial
problem is  solved (in parallel) using Firedrake and (backward Euler) time integration is handled by PyMGRIT.
As usual, for PyMGRIT the following is required:

* `Vector class`_
* `Application class`_
* `Example run`_

Vector class
^^^^^^^^^^^^
::

    try:
        from firedrake import norm, Function
    except ImportError as e:
        import sys

        sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

    import numpy as np

    from pymgrit.core.vector import Vector


    class VectorFiredrake(Vector):
        """
        Vector class for Firedrake Function object
        """

        def __init__(self, values: Function):
            """
            Constructor.
            """

            super().__init__()
            if isinstance(values, Function):
                self.values = values.copy(deepcopy=True)
            else:
                raise Exception('Wrong datatype')

        def set_values(self, values):
            """
            Set vector data

            :param values: values for vector object
            """
            self.values = values

        def get_values(self):
            """
            Get vector data

            :return: values of vector object
            """
            return self.values

        def clone(self):
            """
            Initialize vector object with copied values

            :rtype: vector object with zero values
            """
            return VectorFiredrake(self.values)

        def clone_zero(self):
            """
            Initialize vector object with zeros

            :rtype: vector object with zero values
            """
            tmp = VectorFiredrake(self.values)
            tmp = tmp * 0
            return tmp

        def clone_rand(self):
            """
            Initialize vector object with random values

            :rtype: vector object with random values
            """
            tmp = VectorFiredrake(self.values)
            tmp_values = tmp.get_values()
            tmp_values.dat.data[:] = np.random.rand(len(tmp_values.dat.data[:]))
            tmp.set_values(tmp_values)
            return tmp

        def __add__(self, other):
            """
            Addition of two vector objects (self and other)

            :param other: vector object to be added to self
            :return: sum of vector object self and input object other
            """
            tmp = VectorFiredrake(self.values)
            tmp_value = tmp.get_values()
            tmp_value.dat += other.get_values().dat
            tmp.set_values(tmp_value)
            return tmp

        def __sub__(self, other):
            """
            Subtraction of two vector objects (self and other)

            :param other: vector object to be subtracted from self
            :return: difference of vector object self and input object other
            """
            tmp = VectorFiredrake(self.values)
            tmp_value = tmp.get_values()
            tmp_value.dat -= other.get_values().dat
            tmp.set_values(tmp_value)
            return tmp

        def __mul__(self, other):
            """
            Multiplication of a vector object and a float (self and other)

            :param other: object to be multiplied with self
            :return: difference of vector object self and input object other
            """
            tmp = VectorFiredrake(self.values)
            tmp_value = tmp.get_values()
            tmp_value.dat *= other
            tmp.set_values(tmp_value)
            return tmp

        def norm(self):
            """
            Norm of a vector object

            :return: 2-norm of vector object
            """
            return norm(self.values)

        def unpack(self, values):
            """
            Unpack and set data

            :param values: values for vector object
            """
            self.values.dat.data[:] = values

        def pack(self):
            """
            Pack data

            :return: values of vector object
            """
            return self.values.dat.data[:]


Application class
^^^^^^^^^^^^^^^^^

::

    try:
        from firedrake import FunctionSpace, Constant, TestFunction, TrialFunction, Function, FacetNormal, inner, dx, grad
        from firedrake import outer, LinearVariationalProblem, NonlinearVariationalSolver, dS, exp, SpatialCoordinate, avg
    except ImportError as e:
        import sys

        sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

    from mpi4py import MPI

    from pymgrit.core.application import Application
    from pymgrit.firedrake.vector_firedrake import VectorFiredrake


    class Diffusion2D(Application):
        """
        Application class containing the description of the diffusion problem.

        The spatial discretisation is P1 DG (piecewise linear discontinous
        elements) and uses an interior penalty method which penalises jumps
        at element interfaces.
        """

        def __init__(self, mesh: object, kappa: float, comm_space: MPI.Comm, mu: float = 5., *args, **kwargs):
            """
            Constructor

            :param mesh: spatial domain
            :param kappa: diffusion coefficient
            :param mu: penalty weighting function
            """
            super().__init__(*args, **kwargs)

            # Spatial domain and function space
            self.mesh = mesh
            self.function_space = FunctionSpace(self.mesh, "DG", 1)
            self.comm_space = comm_space

            # Placeholder for time step - will be updated in the update method
            self.dt = Constant(0.)

            # Things we need for the form
            gamma = TestFunction(self.function_space)
            phi = TrialFunction(self.function_space)
            self.f = Function(self.function_space)
            n = FacetNormal(mesh)

            # Set up the rhs and bilinear form of the equation
            a = (inner(gamma, phi) * dx
                 + self.dt * (
                         inner(grad(gamma), grad(phi) * kappa) * dx
                         - inner(2 * avg(outer(phi, n)), avg(grad(gamma) * kappa)) * dS
                         - inner(avg(grad(phi) * kappa), 2 * avg(outer(gamma, n))) * dS
                         + mu * inner(2 * avg(outer(phi, n)), 2 * avg(outer(gamma, n) * kappa)) * dS
                 )
                 )
            rhs = inner(gamma, self.f) * dx

            # Function to hold the solution
            self.soln = Function(self.function_space)

            # Setup problem and solver
            prob = LinearVariationalProblem(a, rhs, self.soln)
            self.solver = NonlinearVariationalSolver(prob)

            # Set the data structure for any user-defined time point
            self.vector_template = VectorFiredrake(self.soln)

            # Set initial condition:
            # Setting up a Gaussian blob in the centre of the domain.
            x = SpatialCoordinate(self.mesh)
            initial_tracer = exp(-((x[0] - 5) ** 2 + (x[1] - 5) ** 2))
            tmp = Function(self.function_space)
            tmp.interpolate(initial_tracer)
            self.vector_t_start = VectorFiredrake(tmp)

        def step(self, u_start: VectorFiredrake, t_start: float, t_stop: float) -> VectorFiredrake:
            """
            Time integration routine for 2D diffusion problem:
                Backward Euler

            :param u_start: approximate solution for the input time t_start
            :param t_start: time associated with the input approximate solution u_start
            :param t_stop: time to evolve the input approximate solution to
            :return: approximate solution at input time t_stop
            """
            # Time-step size
            self.dt.assign(t_stop - t_start)

            self.f.assign(u_start.get_values())

            # Take Backward Euler step
            self.solver.solve()

            return VectorFiredrake(self.soln)


Example run
^^^^^^^^^^^

::

    try:
        from firedrake import PeriodicSquareMesh
    except ImportError as e:
        import sys

        sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

    from mpi4py import MPI

    from pymgrit.core.mgrit import Mgrit
    from pymgrit.core.split import split_communicator
    from pymgrit.firedrake.diffusion_2d_firedrake import Diffusion2D

    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 1)

    # Define spatial domain
    n = 20
    mesh = PeriodicSquareMesh(n, n, 10, comm=comm_x)

    # Set up the problem
    diffusion0 = Diffusion2D(mesh=mesh, kappa=0.1, comm_space=comm_x, t_start=0, t_stop=10, nt=17)
    diffusion1 = Diffusion2D(mesh=mesh, kappa=0.1, comm_space=comm_x, t_start=0, t_stop=10, nt=9)

    # Setup three-level MGRIT solver with the space and time communicators and
    # solve the problem
    mgrit = Mgrit(problem=[diffusion0, diffusion1], comm_time=comm_t, comm_space=comm_x)
    info = mgrit.solve()

