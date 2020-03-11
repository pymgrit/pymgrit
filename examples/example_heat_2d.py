import pathlib
import numpy as np

from pymgrit.heat.heat_2d import Heat2D
from pymgrit.core.mgrit import Mgrit


def main():
    def output_fcn(self):
        name = 'heat_equation_2d'
        pathlib.Path('results/' + name + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 't': self.problem[0].t}
        np.save('results/' + name + '/' + str(self.solve_iter) + '/heat_equation_2d', sol)

    nx = 123
    ny = 58
    t_start = 0.0
    t_stop = 5.5

    heat0 = Heat2D(lx=0.75, ly=1.5, nx=nx, ny=ny, a=3.5, t_start=t_start, t_stop=t_stop, nt=21)
    heat1 = Heat2D(lx=0.75, ly=1.5, nx=nx, ny=ny, a=3.5, t_interval=heat0.t[::2])
    heat2 = Heat2D(lx=0.75, ly=1.5, nx=nx, ny=ny, a=3.5, t_interval=heat1.t[::2])

    problem = [heat0, heat1, heat2]
    mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='V', nested_iteration=False, max_iter=10,
                  output_fcn=output_fcn, output_lvl=2, logging_lvl=20, random_init_guess=False)

    info = mgrit.solve()

    u = np.load(
        'results/heat_equation_2d/' + str(len(info['conv'])) + '/heat_equation_2d.npy',
        allow_pickle=True).item()['u']
    for i in range(len(u)):
        u_e = heat0.u_exact(x=heat0.x[:, np.newaxis], y=heat0.y[np.newaxis, :], t=heat0.t[i])
        diff = abs(u[i].get_values() - u_e[1:-1, 1:-1]).max()
        print("{0:.5f}".format(heat0.t[i]), ': ', diff)


if __name__ == '__main__':
    main()
