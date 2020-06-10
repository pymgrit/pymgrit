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

`example_firedrake_diffusion_2d.py`_

.. _example_firedrake_diffusion_2d.py: https://github.com/pymgrit/pymgrit/blob/master/examples/firedrake/example_firedrake_diffusion_2d.py

The following example shows how to set up and solve a 2D diffusion problem using PyMGRIT and Firedrake. The spatial
problem is  solved (in parallel) using Firedrake and (backward Euler) time integration is handled by PyMGRIT.
As usual, for PyMGRIT the following is required:

* `Vector class`_
* `Application class`_
* `Example run`_

Vector class
^^^^^^^^^^^^
::

    import numpy as np

    from mpi4py import MPI

    from pymgrit.core.mgrit import Mgrit
    from pymgrit.core.split import split_communicator
    from pymgrit.core.application import Application
    from pymgrit.core.vector import Vector

    from firedrake import PeriodicSquareMesh
    from firedrake import FunctionSpace, Constant, TestFunction, TrialFunction, Function, FacetNormal, inner, dx, grad, avg
    from firedrake import outer, LinearVariationalProblem, NonlinearVariationalSolver, dS, exp, SpatialCoordinate


    class VectorDiffusion2D(Vector):
        """
        Vector class for the 2D diffusion equation

        Note: Vector objects only hold the values of all spatial degrees of
              freedom associated with a time point. Firedrake related data
              is saved in an object of the Diffusion2D application class.
        """

        def __init__(self, size: int, comm_space: MPI.Comm):
            """
            Constructor.

            :param size: number of degrees of freedom in spatial domain
            """

            super().__init__()
            self.size = size
            self.values = np.zeros(size)
            self.comm_space = comm_space

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
            tmp = VectorDiffusion2D(size=self.size, comm_space=self.comm_space)
            tmp.set_values(self.get_values())
            return tmp

        def clone_zero(self):
            """
            Initialize vector object with zeros

            :rtype: vector object with zero values
            """
            return VectorDiffusion2D(size=self.size, comm_space=self.comm_space)

        def clone_rand(self):
            """
            Initialize vector object with random values

            :rtype: vector object with random values
            """
            tmp = VectorDiffusion2D(size=self.size, comm_space=self.comm_space)
            tmp.set_values(np.random.rand(self.size))
            return tmp

        def __add__(self, other):
            """
            Addition of two vector objects (self and other)

            :param other: vector object to be added to self
            :return: sum of vector object self and input object other
            """
            tmp = VectorDiffusion2D(self.size, comm_space=self.comm_space)
            tmp.set_values(self.get_values() + other.get_values())
            return tmp

        def __sub__(self, other):
            """
            Subtraction of two vector objects (self and other)

            :param other: vector object to be subtracted from self
            :return: difference of vector object self and input object other
            """
            tmp = VectorDiffusion2D(self.size, comm_space=self.comm_space)
            tmp.set_values(self.get_values() - other.get_values())
            return tmp

        def norm(self):
            """
            Norm of a vector object

            :return: 2-norm of vector object
            """
            tmp = self.comm_space.allgather(self.values)
            return np.linalg.norm(np.array([item for sublist in tmp for item in sublist]))

        def unpack(self, values):
            """
            Unpack and set data

            :param values: values for vector object
            """
            self.values = values

        def pack(self):
            """
            Pack data

            :return: values of vector object
            """
            return self.values


Application class
^^^^^^^^^^^^^^^^^

::

    class Diffusion2D(Application):
    """
    Application class containing the description of the diffusion problem.

    The spatial domain is a 10x10 square with
    periodic boundary conditions in each direction.

    The initial condition is a Gaussian in the centre of the domain.

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
            V = FunctionSpace(self.mesh, "DG", 1)
            self.function_space = V
            self.comm_space = comm_space

            # Placeholder for time step - will be updated in the update method
            self.dt = Constant(0.)

            # Things we need for the form
            gamma = TestFunction(V)
            phi = TrialFunction(V)
            self.f = Function(V)
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
            self.soln = Function(V)

            # Setup problem and solver
            prob = LinearVariationalProblem(a, rhs, self.soln)
            self.solver = NonlinearVariationalSolver(prob)

            # Set the data structure for any user-defined time point
            self.vector_template = VectorDiffusion2D(size=len(self.function_space), comm_space=self.comm_space)

            # Set initial condition:
            # Setting up a Gaussian blob in the centre of the domain.
            self.vector_t_start = VectorDiffusion2D(size=len(self.function_space), comm_space=self.comm_space)
            x = SpatialCoordinate(self.mesh)
            initial_tracer = exp(-((x[0] - 5) ** 2 + (x[1] - 5) ** 2))
            tmp = Function(self.function_space)
            tmp.interpolate(initial_tracer)
            self.vector_t_start.set_values(np.copy(tmp.dat.data))

        def step(self, u_start: VectorDiffusion2D, t_start: float, t_stop: float) -> VectorDiffusion2D:
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

            # Get data from VectorDiffusion2D object u_start
            # and copy to Firedrake Function object tmp
            tmp = Function(self.function_space)
            for i in range(len(u_start.values)):
                tmp.dat.data[i] = u_start.values[i]
            self.f.assign(tmp)

            # Take Backward Euler step
            self.solver.solve()

            # Copy data from Firedrake Function object to VectorDiffusion2D object
            ret = VectorDiffusion2D(size=len(self.function_space), comm_space=self.comm_space)
            ret.set_values(np.copy(self.soln.dat.data))

            return ret

Example run
^^^^^^^^^^^

::

    from mpi4py import MPI

    from firedrake import PeriodicSquareMesh
    from pymgrit.core.mgrit import Mgrit
    from pymgrit.core.split import split_communicator

    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 2)

    # Define spatial domain
    # The domain is a 10x10 square with periodic boundary conditions in each direction.
    n = 20
    mesh = PeriodicSquareMesh(n, n, 10, comm=comm_x)

    # Set up the problem
    diffusion0 = Diffusion2D(mesh=mesh, kappa=0.1, comm_space=comm_x, t_start=0, t_stop=10, nt=65)
    diffusion1 = Diffusion2D(mesh=mesh, kappa=0.1, comm_space=comm_x, t_start=0, t_stop=10, nt=17)
    diffusion2 = Diffusion2D(mesh=mesh, kappa=0.1, comm_space=comm_x, t_start=0, t_stop=10, nt=5)

    # Setup three-level MGRIT solver with the space and time communicators and
    # solve the problem
    mgrit = Mgrit(problem=[diffusion0, diffusion1, diffusion2], comm_time=comm_t, comm_space=comm_x)
    info = mgrit.solve()
