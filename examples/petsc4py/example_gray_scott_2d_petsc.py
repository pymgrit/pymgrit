"""
Example demonstrating the coupling with petsc4py.

Apply three-level MGRIT V-cycles with FCF-relaxation to solve a 2D Gray Scott model.

Note: This example requires petsc4py!
"""

from mpi4py import MPI

from pymgrit.core.split import split_communicator
from pymgrit.core.mgrit import Mgrit
from pymgrit.petsc.gray_scott_2d_petsc import GrayScott

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


def main():
    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 4)

    # Create PETSc DMDA grids
    nx = 129
    ny = 129
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_world)

    # Set up the problem
    problem_level_0 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=0.1, nt=33)
    problem_level_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=problem_level_0.t[::2])
    problem_level_2 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=problem_level_0.t[::2])

    # Setup three-level MGRIT solver with the space and time communicators and
    # solve the problem
    mgrit = Mgrit(problem=[problem_level_0, problem_level_1, problem_level_2], comm_time=comm_t, comm_space=comm_x)
    info = mgrit.solve()


if __name__ == '__main__':
    main()
