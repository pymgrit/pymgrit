import numpy as np

from pymgrit.heat.heat_2d import Heat2D
from pymgrit.core.mgrit import Mgrit


def main():
    def rhs(x, y, t):
        return -np.sin(x) * np.sin(y) * (np.sin(t) - 2 * np.cos(t))

    heat0 = Heat2D(x_start=0, x_end=1, y_start=0, y_end=1, nx=101, ny=51, a=1, rhs=rhs, t_start=0, t_stop=5,
                   nt=2 ** 13 + 1)
    mgrit = Mgrit(problem=[heat0]).solve()


if __name__ == '__main__':
    main()
