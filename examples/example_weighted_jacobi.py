"""
Apply five-level MGRIT F-cycles with FCF-relaxation to solve the 1D heat equation
    u_t - a*u_xx = b(x,t),  a > 0, x in [0,1], t in [0,T],
with RHS b(x,t) = -sin(pi*x)(sin(t) - a*pi^2*cos(t)),
homogeneous Dirichlet BCs in space,
    u(0,t)  = u(1,t) = 0,   t in [0,T],
and subject to the initial condition
    u(x,0)  = sin(pi*x),    x in [0,1]

=> exact solution u(x,t) = sin(pi*x)*cos(t)
"""

import numpy as np

from pymgrit.heat.heat_1d import Heat1D
from pymgrit.core.mgrit import Mgrit


def main():
    def rhs(x, t):
        """
        Right-hand side of 1D heat equation example problem at a given space-time point (x,t),
          -sin(pi*x)(sin(t) - a*pi^2*cos(t)),  a = 1

        Note: exact solution is np.sin(np.pi * x) * np.cos(t)
        :param x: spatial grid point
        :param t: time point
        :return: right-hand side of 1D heat equation example problem at point (x,t)
        """

        return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))

    def init_cond(x):
        """
        Initial condition of 1D heat equation example,
          u(x,0)  = sin(pi*x)

        :param x: spatial grid point
        :return: initial condition of 1D heat equation example problem
        """
        return np.sin(np.pi * x)

    heat0 = Heat1D(x_start=0, x_end=1, nx=1001, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=65)
    heat1 = Heat1D(x_start=0, x_end=1, nx=1001, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=33)
    heat2 = Heat1D(x_start=0, x_end=1, nx=1001, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=17)
    heat3 = Heat1D(x_start=0, x_end=1, nx=1001, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=9)
    heat4 = Heat1D(x_start=0, x_end=1, nx=1001, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=5)

    # Setup five-level MGRIT solver and solve the problem
    problem = [heat0, heat1, heat2, heat3, heat4]

    # Unitary C-relaxation weight
    mgrit = Mgrit(problem=problem, tol=1e-8, cf_iter=1, cycle_type='F', nested_iteration=False, max_iter=10,
                  logging_lvl=20, random_init_guess=False)
    info = mgrit.solve()

    # Non-unitary C-relaxation weight
    mgrit2 = Mgrit(problem=problem, weight_c=1.3, tol=1e-8, cf_iter=1, cycle_type='F', nested_iteration=False, max_iter=10,
                  logging_lvl=20, random_init_guess=False)
    info = mgrit2.solve()


if __name__ == '__main__':
    main()
