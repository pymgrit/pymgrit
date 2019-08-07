from firedrake import *
import numpy as np

class Diffusion(object):
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

    def __init__(self, n, kappa, mu=5.):

        # mesh and DG function space
        mesh = PeriodicSquareMesh(n, n, 10)
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
        self.soln = Function(V)

        # setup problem and solver
        prob = LinearVariationalProblem(a, L, self.soln)
        self.solver = NonlinearVariationalSolver(prob)

    def initialise(self, f):
        """
        Initialisation function, setting up a Gaussian blob in the
        centre of the domain.
        """

        x = SpatialCoordinate(f.function_space().mesh())
        initial_tracer = exp(-((x[0]-5)**2 + (x[1]-5)**2))
        f.interpolate(initial_tracer)

    def update(self, f_in, f_out, dt):
        """
        Update function, taking in f_in and updating f_out by applying
        backward Euler for a single timestep dt
        """
        self.dt.assign(dt)
        self.f.assign(f_in)
        self.solver.solve()
        f_out.assign(self.soln)


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

    solution = []
    solution.append(np.copy(fnp1.dat.data))

    # this is the timeloop
    while t + 0.5*dt < tmax:

        # compute backward Euler update
        problem.update(fn, fnp1, dt)

        # update function and timestep
        fn.assign(fnp1)
        t += dt

        # write out data
        output.write(fnp1)
        solution.append(np.copy(fnp1.dat.data))

    return solution

if __name__ == '__main__':
    # set up a diffusion problem with dx=0.5 (L=10, n=20) and diffusion
    # coefficient kappa=0.1
    problem = Diffusion(20, 0.1)

    # run the timeloop with dt=0.1 and tmax=10.
    sol = timeloop(problem, 0.1, 10)

    np.save('2D_diffusion', np.vstack(sol))
