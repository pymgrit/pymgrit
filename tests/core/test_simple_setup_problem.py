"""
Tests simple_setup_problem
"""
import numpy as np

from pymgrit.core.vector import Vector
from pymgrit.core.application import Application
from pymgrit.core.simple_setup_problem import simple_setup_problem


class VectorSimple(Vector):
    def __init__(self):
        super(VectorSimple, self).__init__()

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def norm(self):
        pass

    def clone_zero(self):
        pass

    def clone_rand(self):
        pass

    def set_values(self, values):
        pass

    def get_values(self):
        pass

    def pack(self):
        pass

    def unpack(self):
        pass


class ApplicationSimple(Application):
    def __init__(self, *args, **kwargs):
        super(ApplicationSimple, self).__init__(*args, **kwargs)
        self.vector_template = VectorSimple()  # Setting the class which is used for each time point
        self.vector_t_start = VectorSimple()

    def step(self, u_start: VectorSimple, t_start: float, t_stop: float) -> VectorSimple:
        return 1


def test_simple_setup_problem():
    """
    Test simple_setup_problem
    """
    app = ApplicationSimple(t_start=0, t_stop=1, nt=101)

    problem = simple_setup_problem(problem=app, level=3, coarsening=2)

    np.testing.assert_equal(True, isinstance(problem[0], ApplicationSimple))
    np.testing.assert_equal(True, isinstance(problem[1], ApplicationSimple))
    np.testing.assert_equal(True, isinstance(problem[2], ApplicationSimple))

    np.testing.assert_equal(problem[0].nt, 101)
    np.testing.assert_equal(problem[0].t_start, 0)
    np.testing.assert_equal(problem[0].t_end, 1)
    np.testing.assert_equal(problem[0].t, np.linspace(0, 1, 101))

    np.testing.assert_equal(problem[1].nt, 51)
    np.testing.assert_equal(problem[1].t_start, 0)
    np.testing.assert_equal(problem[1].t_end, 1)
    np.testing.assert_equal(problem[1].t, np.linspace(0, 1, 101)[::2])

    np.testing.assert_equal(problem[2].nt, 26)
    np.testing.assert_equal(problem[2].t_start, 0)
    np.testing.assert_equal(problem[2].t_end, 1)
    np.testing.assert_equal(problem[2].t, np.linspace(0, 1, 101)[::4])


def test_simple_setup_problem_warning():
    """
    Test simple_setup_problem
    """
    app = ApplicationSimple(t_start=0, t_stop=1, nt=2)
    problem = simple_setup_problem(problem=app, level=2, coarsening=2)

    np.testing.assert_equal(True, isinstance(problem[0], ApplicationSimple))
    np.testing.assert_equal(True, isinstance(problem[1], ApplicationSimple))

    np.testing.assert_equal(problem[0].nt, 2)
    np.testing.assert_equal(problem[0].t_start, 0)
    np.testing.assert_equal(problem[0].t_end, 1)
    np.testing.assert_equal(problem[0].t, np.linspace(0, 1, 2))

    np.testing.assert_equal(problem[1].nt, 1)
    np.testing.assert_equal(problem[1].t_start, 0)
    np.testing.assert_equal(problem[1].t_end, 0)
    np.testing.assert_equal(problem[1].t, np.linspace(0, 1, 2)[::2])
