from mpi4py import MPI


def split_commworld(comm, px):
    rank = comm.Get_rank()
    xcolor = rank // px
    tcolor = rank % px

    comm_x = comm.Split(color=xcolor, key=rank)
    comm_t = comm.Split(color=tcolor, key=rank)

    return comm_x, comm_t


def test():
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    size = comm_world.Get_size()
    comm_x, comm_t = split_commworld(comm_world, 1)

    rank_x = comm_x.Get_rank()
    size_x = comm_x.Get_size()

    rank_t = comm_t.Get_rank()
    size_t = comm_t.Get_size()

    print('Global', rank, '/', size, 'time', rank_t, '/', size_t, 'space', rank_x, '/', size_x)


if __name__ == '__main__':
    test()
