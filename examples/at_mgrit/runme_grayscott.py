import time

from mpi4py import MPI

from pymgrit.core.at_mgrit import AtMgrit
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.split import split_communicator
from pymgrit.petsc.gray_scott_2d_petsc import GrayScott

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


def run_ts(space_parallel=1):
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, space_parallel)

    nx = 128
    ny = 128
    L = 2.5
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_x)

    Du, Dv, F, K = 8.0e-05, 4.0e-05, 0.024, 0.06

    gray_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=256, nt=2 ** 14, L=L,
                       du=Du, dv=Dv, a=F, b=F + K)

    start = time.time()
    value = gray_1.vector_t_start
    for i in range(1, len(gray_1.t)):
        value = gray_1.step(u_start=value, t_start=gray_1.t[i - 1], t_stop=gray_1.t[i])
    if comm_x.Get_rank() == 0:
        print("time:", time.time() - start)


def run_parareal(m0, space_parallel=1):
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, space_parallel)

    nx = 128
    ny = 128
    L = 2.5
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_x)

    Du, Dv, F, K = 8.0e-05, 4.0e-05, 0.024, 0.06

    gray_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=256, nt=2 ** 14, L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_2 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_1.t[::m0], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)

    solver = Mgrit(problem=[gray_1, gray_2], nested_iteration=True, comm_time=comm_t, comm_space=comm_x, cf_iter=0)
    solver.solve()


def run_mgrit(m0, m1, space_parallel=1):
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, space_parallel)

    nx = 128
    ny = 128
    L = 2.5
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_x)

    Du, Dv, F, K = 8.0e-05, 4.0e-05, 0.024, 0.06

    gray_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=256, nt=2 ** 14, L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_2 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_1.t[::m0], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_3 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_2.t[::m1], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)

    solver = Mgrit(problem=[gray_1, gray_2, gray_3], nested_iteration=True, comm_time=comm_t, comm_space=comm_x)
    solver.solve()


def run_at_mgrit_two_level(m0, k, space_parallel=1):
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, space_parallel)

    nx = 128
    ny = 128
    L = 2.5
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_x)

    Du, Dv, F, K = 8.0e-05, 4.0e-05, 0.024, 0.06

    gray_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=256, nt=2 ** 14, L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_2 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_1.t[::m0], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)

    solver = AtMgrit(problem=[gray_1, gray_2], logging_lvl=20, nested_iteration=True, comm_time=comm_t,
                     comm_space=comm_x, k=k, cf_iter=0)
    solver.solve()


def run_at_mgrit_three_level(m0, m1, k, space_parallel=1):
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, space_parallel)

    nx = 128
    ny = 128
    L = 2.5
    da = PETSc.DMDA().create([nx, ny], dof=2, boundary_type=3, stencil_width=1, comm=comm_x)

    Du, Dv, F, K = 8.0e-05, 4.0e-05, 0.024, 0.06

    gray_1 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_start=0, t_stop=256, nt=2 ** 14, L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_2 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_1.t[::m0], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)
    gray_3 = GrayScott(dmda=da, comm_x=comm_x, method='IMPL', t_interval=gray_2.t[::m1], L=L,
                       du=Du, dv=Dv, a=F, b=F + K)

    solver = AtMgrit(problem=[gray_1, gray_2, gray_3], logging_lvl=20, nested_iteration=True, comm_time=comm_t,
                     comm_space=comm_x, k=k)
    solver.solve()


if __name__ == '__main__':
    # run_parareal(m0=512)
    # run_parareal(m0=256)
    # run_parareal(m0=128)
    # run_parareal(m0=64)
    #
    # run_at_mgrit_two_level(m0=512, k=16)
    # run_at_mgrit_two_level(m0=256, k=32)
    # run_at_mgrit_two_level(m0=128, k=64)
    # run_at_mgrit_two_level(m0=64, k=128)
    #
    # run_mgrit(m0=64, m1=16)
    # run_mgrit(m0=64, m1=8)
    # run_mgrit(m0=64, m1=4)
    # run_mgrit(m0=64, m1=2)
    #
    # run_at_mgrit_three_level(m0=64, m1=16, k=8)
    # run_at_mgrit_three_level(m0=64, m1=8, k=16)
    # run_at_mgrit_three_level(m0=64, m1=4, k=32)
    run_at_mgrit_three_level(m0=64, m1=2, k=64)
