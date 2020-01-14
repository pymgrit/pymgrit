"""
Splits a communicator for space-time parallelism
"""

from typing import Tuple

from mpi4py import MPI


def split_communicator(comm: MPI.Comm, splitting: int) -> Tuple[MPI.Comm, MPI.Comm]:
    """
    Splits the communicator
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
