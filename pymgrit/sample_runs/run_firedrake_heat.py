from mpi4py import MPI
from firedrake import PeriodicSquareMesh
from pymgrit.core import mgrit as solver
import diffusion
from pymgrit.core import grid_transfer_copy
import pathlib
import numpy as np
from pymgrit.core import split


def main():
    def output_fcn(self):
        name = 'firedrake_heat'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}
        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

        from firedrake import File, Function
        tmp = Function(self.problem[0].function_space)
        output = File('results/out_in_func' + str(self.comm_time_rank) + '.pvd')
        for i in range(len(self.u[0])):
            for j in range(len(tmp.dat.data)):
                tmp.dat.data[j] = self.u[0][i].vec[j]
            output.write(tmp)

    # mesh and DG function space
    n = 20

    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split.split_commworld(comm_world, 2)

    mesh = PeriodicSquareMesh(n, n, 10, comm=comm_x)

    heat0 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=65)
    heat1 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=17)
    heat2 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=5)

    problem = [heat0, heat1, heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, it=5, comm_time=comm_t,
                         comm_space=comm_x, output_fcn=output_fcn)
    mgrit.solve()


if __name__ == '__main__':
    main()
