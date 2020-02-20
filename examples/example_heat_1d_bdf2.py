import numpy as np
import pathlib

from pymgrit.core.mgrit import Mgrit
from pymgrit.heat.heat_1d_2pts_bdf1 import Heat1DBDF1
from pymgrit.heat.heat_1d_2pts_bdf2 import Heat1DBDF2


def main():
    def output_fcn(self):
        name = 'heat_equation_bdf2'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

    t_stop = 2
    nt = 512
    dt = t_stop / nt
    heat0 = Heat1DBDF1(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_start=0, t_stop=t_stop, nt=nt + 1)
    heat1 = Heat1DBDF1(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_interval=heat0.t[::2])
    heat2 = Heat1DBDF1(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_interval=heat1.t[::2])

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    mgrit.solve()

    heat0 = Heat1DBDF2(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_start=0, t_stop=t_stop, nt=nt + 1)
    heat1 = Heat1DBDF2(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_interval=heat0.t[::2])
    heat2 = Heat1DBDF2(x_start=0, x_end=2, nx=1001, a=1, dt=dt, t_interval=heat1.t[::2])

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem)
    return mgrit.solve()


if __name__ == '__main__':
    main()
