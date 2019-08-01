import numpy as np
from firedrake import *
from abstract_classes import application
from firedrake_heat_equation import vector_standard
from numpy import linalg as la

class Diffusion(application.Application):
    """
    Class containing the description of the diffusion problem.

    The domain is a 10x10 square with periodic boundary conditions in
    each direction.

    The initial condition is a Gaussian in the centre of the domain.

    The spatial discretisation is P1 DG (piecewise linear discontinous
    elements) and uses an interior penalty method which penalises jumps
    at element interfaces.

    n: number of cells in each direction
    kappa: the diffusion coefficient
    mu: the penalty weighting function
    """

    def __init__(self, mesh, kappa, mu=5., *args, **kwargs):

        super(Diffusion, self).__init__(*args, **kwargs)

        self.mesh = mesh
        V = FunctionSpace(self.mesh, "DG", 1)
        self.function_space = V

        # placeholder for timestep - will be updated in the update method
        self.dt = Constant(0.)

        # things we need for the form
        gamma = TestFunction(V)
        phi = TrialFunction(V)
        self.f = Function(V)
        n = FacetNormal(mesh)

        # set up the rhs and lhs of the equation
        a = (
            inner(gamma, phi)*dx
            + self.dt*(
                inner(grad(gamma), grad(phi)*kappa)*dx
                - inner(2*avg(outer(phi, n)), avg(grad(gamma)*kappa))*dS
                - inner(avg(grad(phi)*kappa), 2*avg(outer(gamma, n)))*dS
                + mu*inner(2*avg(outer(phi, n)), 2*avg(outer(gamma, n)*kappa))*dS
                )
        )
        L = inner(gamma, self.f)*dx

        # function to hold the solution
        self.soln = Function(V)

        # setup problem and solver
        prob = LinearVariationalProblem(a, L, self.soln)
        self.solver = NonlinearVariationalSolver(prob)

        # set initial condition
        self.u = vector_standard.VectorStandard(len(self.function_space))
        self.initialise()

    def step(self, u_start, t_start, t_stop):
        # compute backward Euler update
        self.dt.assign(t_stop-t_start)
        tmp = Function(self.function_space)
        for i in range(len(u_start.vec)):
            tmp.dat.data[i] = u_start.vec[i]
        self.f.assign(tmp)
        self.solver.solve()
        f_out = vector_standard.VectorStandard(len(self.function_space))
        f_out.vec = np.copy(self.soln.dat.data)

        return f_out

    def initialise(self):
        """
        Initialisation function, setting up a Gaussian blob in the
        centre of the domain.
        """

        x = SpatialCoordinate(self.mesh)
        initial_tracer = exp(-((x[0]-5)**2 + (x[1]-5)**2))
        tmp = Function(self.function_space)
        tmp.interpolate(initial_tracer)
        self.u.vec = np.copy(tmp.dat.data)

