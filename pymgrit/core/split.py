"""
Splits a communicator for space-time parallelism
"""

from typing import Tuple

from mpi4py import MPI


def split_communicator(comm: MPI.Comm, splitting: int) -> Tuple[MPI.Comm, MPI.Comm]:
    """
    Splits the
    :param comm:
    :param splitting:
    :return:
    """
    rank = comm.Get_rank()
    x_color = rank // splitting
    t_color = rank % splitting

    comm_x = comm.Split(color=x_color, key=rank)
    comm_t = comm.Split(color=t_color, key=rank)

    return comm_x, comm_t


# def test_splitting():
#     """
#     Test the splitting.
#     """
#     comm_world = MPI.COMM_WORLD
#     rank = comm_world.Get_rank()
#     size = comm_world.Get_size()
#     comm_x, comm_t = split_communicator(comm_world, 1)
#
#     rank_x = comm_x.Get_rank()
#     size_x = comm_x.Get_size()
#
#     rank_t = comm_t.Get_rank()
#     size_t = comm_t.Get_size()
#
#     print('Global', rank, '/', size, 'time', rank_t, '/', size_t, 'space', rank_x, '/', size_x)
#
#
# if __name__ == '__main__':
#     test_splitting()
