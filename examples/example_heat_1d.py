"""
Apply five-level MGRIT F-cycles with FCF-relaxation to solve the 1D heat equation
"""

from pymgrit.heat.heat_1d import Heat1D
from pymgrit.core.mgrit import Mgrit


def main():
    heat0 = Heat1D(x_start=0, x_end=2, nx=1001, a=1, t_start=0, t_stop=2, nt=65)
    heat1 = Heat1D(x_start=0, x_end=2, nx=1001, a=1, t_start=0, t_stop=2, nt=33)
    heat2 = Heat1D(x_start=0, x_end=2, nx=1001, a=1, t_start=0, t_stop=2, nt=17)
    heat3 = Heat1D(x_start=0, x_end=2, nx=1001, a=1, t_start=0, t_stop=2, nt=9)
    heat4 = Heat1D(x_start=0, x_end=2, nx=1001, a=1, t_start=0, t_stop=2, nt=5)

    problem = [heat0, heat1, heat2, heat3, heat4]
    mgrit = Mgrit(problem=problem, cf_iter=1, cycle_type='F', nested_iteration=False, max_iter=10,
                  logging_lvl=20, random_init_guess=False)

    info = mgrit.solve()


if __name__ == '__main__':
    main()
