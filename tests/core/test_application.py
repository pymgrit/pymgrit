"""
Tests for the application class
"""
import pytest
import numpy as np

from pymgrit.core.application import Application


class ApplicationTest(Application):
    """
    Problem for test the parallel model.
    """

    def __init__(self, *args, **kwargs):
        super(ApplicationTest, self).__init__(*args, **kwargs)
        self.vector_template = 2
        self.vector_t_start = 2

    def step(self, u_start, t_start: float, t_stop: float):
        pass


class ApplicationTest2(Application):
    """
    Problem for test the parallel model.
    """

    def __init__(self, *args, **kwargs):
        super(ApplicationTest2, self).__init__(*args, **kwargs)

    def step(self, u_start, t_start: float, t_stop: float):
        pass


def test_application_constructor1():
    """
    Test the constructor
    """
    a = ApplicationTest(t_start=0, t_stop=1, nt=11)
    result = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=float)
    np.testing.assert_almost_equal(a.t, result)


def test_application_constructor2():
    """
    Test the constructor
    """
    a = ApplicationTest(t_interval=np.linspace(0, 1, 11))
    result = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=float)
    np.testing.assert_almost_equal(a.t, result)


def test_application_constructor3():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest()


def test_application_constructor4():
    """
    Test the constructor
    """
    a = ApplicationTest(t_interval=np.linspace(0, 1, 11), t_start=0)
    result = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=float)
    np.testing.assert_almost_equal(a.t, result)


def test_application_constructor5():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest(t_start=2, t_stop=5)


def test_application_constructor6():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest(nt=4, t_stop=5)


def test_application_constructor7():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest(nt=4, t_start=5)


def test_application_constructor8():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest(t_interval=5)


def test_application_missing_attribute():
    """
    Test the constructor
    """
    with pytest.raises(Exception):
        ApplicationTest2(t_start=0, t_stop=1, nt=11)
