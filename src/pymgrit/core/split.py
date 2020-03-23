"""
Creates new communicators for space & time parallelism
"""

from typing import Tuple

from mpi4py import MPI


def split_communicator(comm: MPI.Comm, splitting: int) -> Tuple[MPI.Comm, MPI.Comm]:
    """
    Creates new communicators for space & time parallelism
    :param comm: Communicator
    :param splitting: Splitting factor
    :return: Space and time communicator
    """

    # Determine color based on splitting factor
    rank = comm.Get_rank()
    x_color = rank // splitting
    t_color = rank % splitting

    # Split the communicator based on the color
    comm_x = comm.Split(color=x_color, key=rank)
    comm_t = comm.Split(color=t_color, key=rank)

    return comm_x, comm_t
