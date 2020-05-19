"""
Test for processes without points
"""
import numpy as np

from pymgrit.dahlquist.dahlquist import Dahlquist
from pymgrit.core.mgrit import Mgrit


def main():
    t_interval = np.linspace(0, 5, 17)
    dahlquist_0 = Dahlquist(t_start=0, t_stop=5, nt=129)
    dahlquist_1 = Dahlquist(t_interval=dahlquist_0.t[::16])
    dahlquist_2 = Dahlquist(t_interval=dahlquist_1.t[::2])
    dahlquist_3 = Dahlquist(t_interval=dahlquist_2.t[::2])
    dahlquist_4 = Dahlquist(t_interval=dahlquist_3.t[::2])
    mgrit = Mgrit(problem=[dahlquist_0, dahlquist_1, dahlquist_2, dahlquist_3,dahlquist_4], tol=1e-10)
    info = mgrit.solve()


if __name__ == '__main__':
    main()
