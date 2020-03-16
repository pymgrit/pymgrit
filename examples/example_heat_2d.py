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

    lx = 0.75
    ly = 1.5
    a = 3.5

    def f(x, y, t):
        return 5 * x * (lx - x) * y * (ly - y) + 10 * a * t * (y * (ly - y) + x * (lx - x))

    heat0 = Heat2D(lx=lx, ly=ly, nx=55, ny=125, a=a, rhs=f, t_start=0, t_stop=1, nt=33)
    heat1 = Heat2D(lx=lx, ly=ly, nx=55, ny=125, a=a, rhs=f, t_interval=heat0.t[::2])

    mgrit = Mgrit(problem=[heat0, heat1], cycle_type='V', output_fcn=output_fcn)

    def u_exact(x, y, t):
        return 5 * t * x * (lx - x) * y * (ly - y)

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
            u_e = u_exact(x=heat0.x_2d, y=heat0.y_2d, t=heat0.t[i])
            diff += abs(sol[i][1].get_values() - u_e).max()
        print("Difference between MGRIT solution and exact solution:", diff)


if __name__ == '__main__':
    main()
