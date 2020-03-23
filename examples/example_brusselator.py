"""
Apply two-level MGRIT with FCF-relaxation to solve Brusselator system
"""

from pymgrit.brusselator.brusselator import Brusselator
from pymgrit.core.mgrit import Mgrit


def main():
    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], cf_iter=1)

    # Solve Brusselator system
    info = mgrit.solve()


if __name__ == '__main__':
    main()
