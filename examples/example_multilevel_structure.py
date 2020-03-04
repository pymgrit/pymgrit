""""
Different ways for building the time-multigrid hiearchy
"""

import numpy as np
from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.simple_setup_problem import simple_setup_problem
from pymgrit.core.mgrit import Mgrit


def main():

    # Option 1: Use PyMGRIT's core function simple_setup_problem()
    dahlquist_multilevel_structure_1 = simple_setup_problem(problem=Dahlquist(t_start=0, t_stop=5, nt=101), level=3,
                                                            coarsening=2)
    Mgrit(problem=dahlquist_multilevel_structure_1, tol=1e-10).solve()

    # Option 2: Build each level using t_start, t_end, and nt
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_start=0, t_stop=5, nt=51)
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_2 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_2, tol=1e-10).solve()

    # Option 3: Specify time intervals for each grid level
    t_interval = np.linspace(0, 5, 101)
    dahlquist_lvl_0 = Dahlquist(t_interval=t_interval)
    dahlquist_lvl_1 = Dahlquist(t_interval=t_interval[::2])  # Takes every second point from t_interval
    dahlquist_lvl_2 = Dahlquist(t_interval=t_interval[::4])  # Takes every fourth point from t_interval
    dahlquist_multilevel_structure_3 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_3, tol=1e-10).solve()

    # Option 4: Mix options 2 and 3
    dahlquist_lvl_0 = Dahlquist(t_start=0, t_stop=5, nt=101)
    dahlquist_lvl_1 = Dahlquist(t_interval=dahlquist_lvl_0.t[::2])  # Using t from the upper level.
    dahlquist_lvl_2 = Dahlquist(t_start=0, t_stop=5, nt=26)
    dahlquist_multilevel_structure_4 = [dahlquist_lvl_0, dahlquist_lvl_1, dahlquist_lvl_2]
    Mgrit(problem=dahlquist_multilevel_structure_4, tol=1e-10).solve()


if __name__ == '__main__':
    main()
