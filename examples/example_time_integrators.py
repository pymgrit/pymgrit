"""
Different time integrators per MGRIT level
"""

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.mgrit import Mgrit


def main():
    # Create Dahlquist's test problem choosing implicit mid-point rule as time stepper
    dahlquist_lvl0 = Dahlquist(t_start=0, t_stop=5, nt=101, method='MR')
    # Create Dahlquist's test problem choosing implicit backward euler as time stepper
    dahlquist_lvl1 = Dahlquist(t_start=0, t_stop=5, nt=51, method='BE')

    # Setup MGRIT and solve the problem
    mgrit = Mgrit(problem=[dahlquist_lvl0, dahlquist_lvl1])
    info = mgrit.solve()


if __name__ == '__main__':
    main()
