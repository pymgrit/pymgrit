import pathlib
import numpy as np
import os

from mpi4py import MPI

from pymgrit.heat.heat_2d import Heat2D
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/heat_equation_2d'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution with corresponding time point to file
        np.save(path + '/heat_equation_2d-rank' + str(self.comm_time_rank),
                [[[self.t[0][i], self.u[0][i]] for i in self.index_local[0]]])

    heat0 = Heat2D(lx=0.75, ly=1.5, nx=125, ny=65, a=3.5, t_start=0, t_stop=5, nt=33)
    heat1 = Heat2D(lx=0.75, ly=1.5, nx=125, ny=65, a=3.5, t_interval=heat0.t[::2])
    heat2 = Heat2D(lx=0.75, ly=1.5, nx=125, ny=65, a=3.5, t_interval=heat1.t[::2])

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='V', nested_iteration=False, max_iter=10,
                  output_fcn=output_fcn, logging_lvl=20, random_init_guess=False)

    info = mgrit.solve()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sol = []
        path = 'results/heat_equation_2d/'
        for filename in os.listdir(path):
            data = np.load(path + filename, allow_pickle=True).tolist()[0]
            sol += data
        sol.sort(key=lambda tup: tup[0])

        diff = 0
        for i in range(len(sol)):
            u_e = heat0.u_exact(x=heat0.x[:, np.newaxis], y=heat0.y[np.newaxis, :], t=heat0.t[i])
            diff += abs(sol[i][1].get_values() - u_e[1:-1, 1:-1]).max()
        print("Difference between MGRIT solution and exact solution:", diff)


if __name__ == '__main__':
    main()
