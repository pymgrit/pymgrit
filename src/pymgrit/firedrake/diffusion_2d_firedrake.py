"""
Application class for the 2D heat equation
"""

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
