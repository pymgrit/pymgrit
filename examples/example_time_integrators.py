"""
Time-grid hierarchy with different time integrators
"""

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.mgrit import Mgrit


def main():
    # Create Dahlquist's test problem using implicit mid-point rule time integration
    dahlquist_lvl0 = Dahlquist(t_start=0, t_stop=5, nt=101, method='MR')
    # Create Dahlquist's test problem using backward Euler time integration
    dahlquist_lvl1 = Dahlquist(t_start=0, t_stop=5, nt=51, method='BE')

    # Setup an MGRIT solver and solve the problem
    mgrit = Mgrit(problem=[dahlquist_lvl0, dahlquist_lvl1])
    info = mgrit.solve()


if __name__ == '__main__':
    main()
