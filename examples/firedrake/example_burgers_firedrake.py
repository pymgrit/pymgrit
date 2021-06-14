"""
Example demonstrating the coupling with Firedrake.

Apply three-level MGRIT V-cycles with FCF-relaxation to solve a 2D burgers equation.

Note: This example requires Firedrake!
      See https://www.firedrakeproject.org for more information on Firedrake.
"""

try:
    from firedrake import UnitSquareMesh, PeriodicIntervalMesh
except ImportError as e:
    import sys

    sys.exit("This example requires firedrake. See https://pymgrit.github.io/pymgrit/coupling/firedrake.html")

from mpi4py import MPI

from pymgrit.firedrake.burgers_firedrake import Burgers1D, Burgers2D
from pymgrit.core.split import split_communicator
from pymgrit.core.mgrit import Mgrit


def main_1d():
    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 2)

    # Define spatial domain
    mesh = PeriodicIntervalMesh(100, length=2, comm=comm_x)

    # Set up the problem
    problem_level_0 = Burgers1D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=129)
    problem_level_1 = Burgers1D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=65)
    problem_level_2 = Burgers1D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=33)
    mgrit = Mgrit(problem=[problem_level_0, problem_level_1, problem_level_2], comm_time=comm_t, comm_space=comm_x)
    info = mgrit.solve()


def main_2d():
    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 2)

    # Define spatial domain
    mesh = UnitSquareMesh(30, 30, comm=comm_x)

    # Set up the problem
    problem_level_0 = Burgers2D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=129)
    problem_level_1 = Burgers2D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=65)
    problem_level_2 = Burgers2D(mesh=mesh, nu=1e-2, comm_space=comm_x, t_start=0, t_stop=0.5, nt=33)
    mgrit = Mgrit(problem=[problem_level_0, problem_level_1, problem_level_2], comm_time=comm_t, comm_space=comm_x)
    info = mgrit.solve()


if __name__ == '__main__':
    main_1d()
    # main_2d()
