"""
Apply three-level MGRIT V-cycles with FCF-relaxation to solve the 1D heat equation
    u_t - a*u_xx = b(x,t),  a > 0, x in [0,1], t in [0,T],
with RHS b(x,t) = -sin(pi*x)(sin(t) - a*pi^2*cos(t)),
homogeneous Dirichlet BCs in space,
    u(0,t)  = u(1,t) = 0,   t in [0,T],
and subject to the initial condition
    u(x,0)  = sin(pi*x),    x in [0,1]

=> exact solution u(x,t) = sin(pi*x)*cos(t)
"""

import numpy as np

from pymgrit.core.mgrit import Mgrit
from pymgrit.heat.heat_1d_2pts_bdf1 import Heat1DBDF1
from pymgrit.heat.heat_1d_2pts_bdf2 import Heat1DBDF2


def main():
    def rhs(x, t):
        """
        Right-hand side of 1D heat equation example problem at a given space-time point (x,t),
          -sin(pi*x)(sin(t) - a*pi^2*cos(t)),  a = 1

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

    def exact_sol(x, t):
        """
        Exact solution of 1D heat equation example problem at a given space-time point (x,t)
        Note: Can be used for computing the error of the MGRIT approximation

        :param x: spatial grid point
        :param t: time point
        :return: exact solution of 1D heat equation example problem at point (x,t)
        """
        return np.sin(np.pi * x) * np.cos(t)

    # Time interval
    t_start = 0
    t_stop = 2
    nt = 512  # number of time points excluding t_start
    dtau = t_stop / nt  # time-step size

    # Time points are grouped in pairs of two consecutive time points
    #   => (nt/2) + 1 pairs
    # Note: * Each pair is associated with the time value of its first point.
    #       * The second value of the last pair (associated with t_stop) is not used.
    #       * The spacing within each pair is the same (= dt) on all grid levels.
    t_interval = np.linspace(t_start, t_stop, int(nt / 2 + 1))

    heat0 = Heat1DBDF2(x_start=0, x_end=1, nx=1001, a=1, dtau=dtau, rhs=rhs, init_cond=init_cond,
                       t_interval=t_interval)
    heat1 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dtau=dtau, rhs=rhs, init_cond=init_cond,
                       t_interval=heat0.t[::2])
    heat2 = Heat1DBDF1(x_start=0, x_end=1, nx=1001, a=1, dtau=dtau, rhs=rhs, init_cond=init_cond,
                       t_interval=heat1.t[::2])

    # Setup three-level MGRIT solver and solve the problem
    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    info = mgrit.solve()


if __name__ == '__main__':
    main()
