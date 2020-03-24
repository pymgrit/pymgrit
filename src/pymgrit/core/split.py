"""
Creates new communicators for space & time parallelism
"""

from typing import Tuple

from mpi4py import MPI


def split_communicator(comm: MPI.Comm, splitting: int) -> Tuple[MPI.Comm, MPI.Comm]:
    """
    Creates new communicators for space & time parallelism by
    "splitting" the input communicator into two sub-communicators.

    :param comm: Communicator to be used as the basis for new communicators
    :param splitting: Splitting factor (number of processes for spatial parallelism)
    :return: Space and time communicator
    """

    # Determine color based on splitting factor
    # All processes with the same color will be assigned to the same communicator.
    rank = comm.Get_rank()
    x_color = rank // splitting
    t_color = rank % splitting

    # Split the communicator based on the color and key
    comm_x = comm.Split(color=x_color, key=rank)
    comm_t = comm.Split(color=t_color, key=rank)

    return comm_x, comm_t
