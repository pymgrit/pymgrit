import numpy as np

from pymgrit.heat.heat_2d import Heat2D
from pymgrit.core.mgrit import Mgrit


def main():
    def rhs(x, y, t):
        return -np.sin(x) * np.sin(y) * (np.sin(t) - 2 * np.cos(t))

    heat0 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_start=0, t_stop=5,
                   nt=2 ** 13 + 1)
    heat1 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_interval=heat0.t[::8])
    heat2 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_interval=heat1.t[::4])
    heat3 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_interval=heat2.t[::4])
    heat4 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_interval=heat3.t[::4])

    mgrit = Mgrit(problem=[heat0, heat1, heat2, heat3, heat4]).solve()


if __name__ == '__main__':
    main()
