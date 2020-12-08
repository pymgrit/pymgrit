"""
Example demonstrating the coupling with Firedrake.

Apply three-level MGRIT V-cycles with FCF-relaxation to solve a 2D diffusion problem.

Note: This example requires Firedrake!
      See https://www.firedrakeproject.org for more information on Firedrake.
"""

try:
    from firedrake import PeriodicSquareMesh
except ImportError as e:
    import sys

    sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

from mpi4py import MPI

from pymgrit.core.mgrit import Mgrit
from pymgrit.core.split import split_communicator
from pymgrit.firedrake.diffusion_2d_firedrake import Diffusion2D

def main():
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


if __name__ == '__main__':
    main()
