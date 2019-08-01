from firedrake import *
from abstract_classes import application
from firedrake_heat_equation.vector_standard import Vector


class HeatEquation(application.Application):
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

    def __init__(self, mesh, kappa, mu=5., t_start=0, t_stop=0, nt=0):

        super().__init__(t_start, t_stop, nt)

        self.mesh = mesh
        V = FunctionSpace(mesh, "DG", 1)
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
        self.soln = Vector(V)

        # setup problem and solver
        prob = LinearVariationalProblem(a, L, self.soln.vec)
        self.solver = NonlinearVariationalSolver(prob)

        self.u = Vector(V)
        self.initialise()

    def initialise(self):
        """
        Initialisation function, setting up a Gaussian blob in the
        centre of the domain.
        """

        x = SpatialCoordinate(self.mesh)
        initial_tracer = exp(-((x[0]-5)**2 + (x[1]-5)**2))
        self.u.vec.interpolate(initial_tracer)

    def step(self, u_start, t_start, t_stop):
        """
        Update function, taking in f_in and updating f_out by applying
        backward Euler for a single timestep dt
        """
        self.dt.assign(t_stop-t_start)
        self.f.assign(u_start.vec)
        self.solver.solve()
        return self.soln


def timeloop(problem, dt, tmax):
    """
    Timeloop function

    problem: a class containing the spatial discretisation of the problem.
    Must have an initialisation method and an update method corresponding
    to the timestep.
    """

    # setup timeloop
    t = 0
    fn = Function(problem.function_space)
    fnp1 = Function(problem.function_space)

    # initialise problem
    problem.initialise(fn)

    # setup output and write out initial condition
    output = File("out.pvd")
    fnp1.assign(fn)
    output.write(fnp1)


    # this is the timeloop
    while t + 0.5*dt < tmax:

        # compute backward Euler update
        problem.update(fn, fnp1, dt)

        # update function and timestep
        fn.assign(fnp1)
        t += dt

        # write out data
        output.write(fnp1)

if __name__ == '__main__':
    # set up a diffusion problem with dx=0.5 (L=10, n=20) and diffusion
    # coefficient kappa=0.1
    problem = Diffusion(20, 0.1)

    # run the timeloop with dt=0.1 and tmax=10.
    timeloop(problem, 0.1, 10)
