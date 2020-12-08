"""
Application classes for Burgers equation
"""

try:
    from firedrake import FunctionSpace, Constant, TestFunction, TrialFunction, Function, FacetNormal, inner, dx, grad
    from firedrake import outer, LinearVariationalProblem, NonlinearVariationalSolver, dS, exp, SpatialCoordinate, avg
    from firedrake import sin, NonlinearVariationalProblem, pi, VectorFunctionSpace, project, as_vector, dot, nabla_grad
except ImportError as e:
    import sys

    sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

from mpi4py import MPI

from pymgrit.core.application import Application
from pymgrit.firedrake.vector_firedrake import VectorFiredrake


class Burgers1D(Application):
    """
    Application class for the 1d Burgers equation
    """

    def __init__(self, mesh: object, nu: float, comm_space: MPI.Comm, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(*args, **kwargs)

        # Spatial domain and function space
        self.mesh = mesh
        self.nu = nu
        self.comm_space = comm_space

        self.function_space = FunctionSpace(mesh, "Lagrange", 2)

        # Placeholder for time step - will be updated in the update method
        self.dt = Constant(0.)

        # Things we need for the form
        self.u_n1 = Function(self.function_space)
        self.u_n = Function(self.function_space)
        v = TestFunction(self.function_space)

        f = (((self.u_n1 - self.u_n) / self.dt) * v +
             self.u_n1 * self.u_n1.dx(0) * v +
             nu * self.u_n1.dx(0) * v.dx(0)) * dx

        # Setup problem and solver
        problem = NonlinearVariationalProblem(f, self.u_n1)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly",
                                                                             "pc_type": "lu"})

        # Set the data structure for any user-defined time point
        self.vector_template = VectorFiredrake(self.u_n1)

        # Set initial condition:
        self.u_n.interpolate(sin(2 * pi * SpatialCoordinate(mesh)[0]))
        self.vector_t_start = VectorFiredrake(self.u_n)

    def step(self, u_start: VectorFiredrake, t_start: float, t_stop: float) -> VectorFiredrake:
        """
        Time integration routine for 1D Burgers problem:
            Backward Euler

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        self.dt.assign(t_stop - t_start)
        self.u_n.assign(u_start.get_values())
        self.solver.solve()
        return VectorFiredrake(self.u_n1)


class Burgers2D(Application):
    """
    Application class for the 2d Burgers equation
    """

    def __init__(self, mesh: object, nu: float, comm_space: MPI.Comm, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(*args, **kwargs)

        # Spatial domain and function space
        self.mesh = mesh
        self.nu = nu
        self.comm_space = comm_space

        self.function_space = VectorFunctionSpace(mesh, "CG", 2)

        # Placeholder for time step - will be updated in the update method
        self.dt = Constant(0.)

        # Things we need for the form
        self.u_n1 = Function(self.function_space)
        self.u_n = Function(self.function_space)
        v = TestFunction(self.function_space)

        f = (inner((self.u_n1 - self.u_n) / self.dt, v) + inner(dot(self.u_n1, nabla_grad(self.u_n1)), v) + nu * inner(
            grad(self.u_n1), grad(v))) * dx

        # Setup problem and solver
        problem = NonlinearVariationalProblem(f, self.u_n1)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "preonly",
                                                                             "pc_type": "lu"})

        # Set the data structure for any user-defined time point
        self.vector_template = VectorFiredrake(self.u_n1)

        # Set initial condition:
        self.u_n.assign(project(as_vector([sin(pi * SpatialCoordinate(mesh)[0]), 0]), self.function_space))
        self.vector_t_start = VectorFiredrake(self.u_n)

    def step(self, u_start: VectorFiredrake, t_start: float, t_stop: float) -> VectorFiredrake:
        """
        Time integration routine for 2D burgers problem:
            Backward Euler

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        self.dt.assign(t_stop - t_start)
        self.u_n.assign(u_start.get_values())
        self.solver.solve()

        return VectorFiredrake(self.u_n1)
