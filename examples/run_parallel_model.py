import pathlib
import numpy as np

from mpi4py import MPI

from pymgrit.parallel_model import parallel_model, mgrit_parallel_model
from pymgrit.core import grid_transfer_copy


def main():
    def output_fcn(self):
        name = 'heat_equation'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.solve_iter) + '/' + str(self.t[0][0]) + ':' + str(self.t[0][-1]),
                sol)

    sleep = 0.01

    heat0 = parallel_model.ParallelModel(sleep=sleep, t_start=0, t_stop=2, nt=2 ** 10 + 1)
    heat1 = parallel_model.ParallelModel(sleep=sleep, t_start=0, t_stop=2, nt=2 ** 8 + 1)
    heat2 = parallel_model.ParallelModel(sleep=sleep, t_start=0, t_stop=2, nt=2 ** 6 + 1)
    heat3 = parallel_model.ParallelModel(sleep=sleep, t_start=0, t_stop=2, nt=2 ** 4 + 1)
    heat4 = parallel_model.ParallelModel(sleep=sleep, t_start=0, t_stop=2, nt=2 ** 2 + 1)

    problem = [heat0, heat1, heat2, heat3, heat4]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = mgrit_parallel_model.MgritParallelModel(problem=problem, transfer=transfer, cf_iter=1, cycle_type='V',
                                                    nested_iteration=False, it=1,
                                                    output_fcn=output_fcn, output_lvl=2, logging_lvl=20)

    mgrit.solve()

    heat0_solves = MPI.COMM_WORLD.allgather(heat0.count_solves)
    heat1_solves = MPI.COMM_WORLD.allgather(heat1.count_solves)
    heat2_solves = MPI.COMM_WORLD.allgather(heat2.count_solves)
    heat3_solves = MPI.COMM_WORLD.allgather(heat3.count_solves)
    heat4_solves = MPI.COMM_WORLD.allgather(heat4.count_solves)

    heat0_runtime = MPI.COMM_WORLD.allgather(heat0.runtime_solves)
    heat1_runtime = MPI.COMM_WORLD.allgather(heat1.runtime_solves)
    heat2_runtime = MPI.COMM_WORLD.allgather(heat2.runtime_solves)
    heat3_runtime = MPI.COMM_WORLD.allgather(heat3.runtime_solves)
    heat4_runtime = MPI.COMM_WORLD.allgather(heat4.runtime_solves)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print('level 0:', np.max(heat0_solves), np.argmax(heat0_solves))
        print('level 1:', np.max(heat1_solves), np.argmax(heat1_solves))
        print('level 2:', np.max(heat2_solves), np.argmax(heat2_solves))
        print('level 3:', np.max(heat3_solves), np.argmax(heat3_solves))
        print('level 4:', np.max(heat4_solves), np.argmax(heat4_solves))
        print('overall max:',
              np.max(heat0_solves) + np.max(heat1_solves) + np.max(heat2_solves) + np.max(heat3_solves) + np.max(
                  heat4_solves))

    # print('overall:', heat0.count_solves+heat1.count_solves+heat2.count_solves+heat3.count_solves+heat4.count_solves)


if __name__ == '__main__':
    main()
