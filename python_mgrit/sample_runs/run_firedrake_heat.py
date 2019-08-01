from mpi4py import MPI
from firedrake import PeriodicSquareMesh, Ensemble, COMM_WORLD, File, Function
from mgrit import mgrit_fas as solver
from firedrake_heat_equation import diffusion
from firedrake_heat_equation import grid_transfer_copy
import pathlib
import numpy as np
import glob
import os

if __name__ == '__main__':
    def output_fcn(self):
        name = 'firedrake_heat'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}
        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

        from firedrake import File, Function
        tmp = Function(self.problem[0].function_space)
        output = File('results/out_in_func'+ str(self.comm_time_rank) +'.pvd')
        for i in range(len(self.u[0])):
            for j in range(len(tmp.dat.data)):
                tmp.dat.data[j] = self.u[0][i].vec[j]
            output.write(tmp)


    # mesh and DG function space
    n = 20
    manager = Ensemble(COMM_WORLD, 1)
    mesh = PeriodicSquareMesh(n, n, 10, comm=manager.comm)

    heat0 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=101)
    heat1 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=17)
    heat2 = diffusion.Diffusion(mesh, kappa=0.1, t_start=0, t_stop=10, nt=5)

    problem = [heat0 , heat1,heat2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, transfer=transfer, it=5, comm_time=manager.ensemble_comm,
                            comm_space=mesh.comm, output_fcn=output_fcn)
    mgrit.solve()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.barrier()

    # if rank == 0:
    #     output = File("results/out.pvd")
    #     sol = []
    #     goal_dir = os.path.join(os.getcwd(), 'results/firedrake_heat')
    #     filelist = glob.glob(os.path.join(goal_dir, '*'))
    #     for infile in sorted(filelist):
    #         a = np.load(infile, allow_pickle=True).item()
    #         sol.append(a['u'])
    #     sol = [item.vec for sublist in sol for item in sublist]
    #     sol = np.vstack(sol)
    #     #print(np.sum(np.abs(sol)**2,axis=-1)**(1./2))
    #     tmp = Function(heat0.function_space)
    #     for i in range(len(sol)):
    #         for j in range(len(tmp.dat.data)):
    #             tmp.dat.data[j] = sol[i,j]
    #         if i == 0:
    #             print(tmp.dat.data)
    #             print(sol[i])
    #         output.write(tmp)
