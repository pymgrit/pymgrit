"""
Apply two-level MGRIT V-cycles with FCF-relaxation to solve the 2D heat equation
    u_t - a(u_xx + u_yy) = b(x,y,t),  a > 0,
        in [x_start, x_end] x [y_start, y_end] x (t_start, t_end],
with RHS b(x,t) = -sin(pi*x)(sin(t) - a*pi^2*cos(t)),
homogeneous Dirichlet BCs in space,
    u(x_start, y) = u(x_end, y) = u(x, y_start) = u(x, y_end) = 0,
and subject to the initial condition
    u(x,0)  = sin(pi*x),    x in [0,1]

=> exact solution u(x,t) = sin(pi*x)*cos(t)
"""

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

    # example problem parameters
    x_end = 0.75
    y_end = 1.5
    a = 3.5

    def rhs(x, y, t):
        """
        Right-hand side of 2D heat equation example problem at a given space-time point (x,y,t),
          5x(x_end - x)y(y_end - y) + 10at(y(y_end-y)) + x(x_end - x),  a = 3.5

        :param x: x-coordinate of spatial grid point
        :param y: y_coordinate of spatial grid point
        :param t: time point
        :return: right-hand side of 2D heat equation example problem at point (x,y,t)
        """

        return 5 * x * (x_end - x) * y * (y_end - y) + 10 * a * t * (y * (y_end - y) + x * (x_end - x))

    def exact_sol(x, y, t):
        """
        Exact solution of 2D heat equation example problem at a given space-time point (x,y,t)
        Note: Can be used for computing the error of the MGRIT approximation

        :param x: x_coordinate of spatial grid point
        :param y: y_coordinate of spatial grid point
        :param t: time point
        :return: exact solution of 2D heat equation example problem at point (x,y,t)
        """
        return 5 * t * x * (x_end - x) * y * (y_end - y)

    heat0 = Heat2D(x_start=0, x_end=x_end, y_start=0, y_end=y_end, nx=55, ny=125, a=a, rhs=rhs, t_start=0, t_stop=1, nt=33)
    heat1 = Heat2D(x_start=0, x_end=x_end, y_start=0, y_end=y_end, nx=55, ny=125, a=a, rhs=rhs, t_interval=heat0.t[::2])

    # Setup two-level MGRIT solver and solve the problem
    mgrit = Mgrit(problem=[heat0, heat1], cycle_type='V', output_fcn=output_fcn)
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
            u_e = exact_sol(x=heat0.x_2d, y=heat0.y_2d, t=heat0.t[i])
            diff += abs(sol[i][1].get_values() - u_e).max()
        print("Difference between MGRIT solution and exact solution:", diff)


if __name__ == '__main__':
    main()
