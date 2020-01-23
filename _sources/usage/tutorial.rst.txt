**********
Tutorial
**********

This tutorial will walk you trough these tasks:

#. Write your first vector structure
#. Write your first application
#. Solve your application problem

-----------------
Vector structure
-----------------

The first step is to write a data structure that holds the solution of one time point. The data structure must inherit
from the class `Vector` laying in the core of PyMGRIT. Our class gets a size as integer and creates a numpy array of
this size afterwards. Furthermore, the functions must override the functions

    - `__add__`: For the addition of two objects of our class
    - `__sub__`: For the substraction of two objects of our class
    - `norm`: The norm of the class
    - `init_zero`: Initialization of the data with zeros
    - `init_rand`: Initialization of the data with random values
::

    import numpy as np
    from pymgrit.core. import vector

    class VectorStandard(vector.Vector):

        def __init__(self, size):
            super(VectorStandard, self).__init__()
            self.size = size
            self.vec = np.zeros(size)

        def __add__(self, other):
            tmp = VectorStandard(self.size)
            tmp.vec = self.vec + other.vec
            return tmp

        def __sub__(self, other):
            tmp = VectorStandard(self.size)
            tmp.vec = self.vec - other.vec
            return tmp

        def norm(self):
            return np.linalg.norm(self.vec)

        def init_zero(self):
            return VectorStandard(self.size)

        def init_rand(self):
            tmp = VectorStandard(self.size)
            tmp.vec = np.random.rand(self.size)
            return tmp


-----------
Application
-----------

TODO!

The next step is to write your first problem application. Therefore, write a class that inherits from `Application`.
The variable `u` has to be specified and set to the class defined in the previous step. Furthermore, the function
`step` has to be specified::

    from pymgrit.core import application

    class ApplicationExample(application.Application):
        def __init__(self, sleep, *args, **kwargs):
            super(ApplicationExample, self).__init__(*args, **kwargs)
            self.u = VectorStandard(1)  # Create initial value solution

        def step(self, u_start: VectorStandard, t_start: float,t_stop: float) -> VectorStandard:
            ret = VectorStandard(1)
            return ret

-----------------
Solve the problem
-----------------

TODO::

    import pymgrit
    problem_lvl_0 = ApplicationExample(t_start=0, t_stop=2, nt=65)
    problem_lvl_1 = ApplicationExample(t_start=0, t_stop=2, nt=17)
    problem_lvl_2 = ApplicationExample(t_start=0, t_stop=2, nt=5)
    problem = [problem_lvl_0, problem_lvl_1, problem_lvl_2]
    mgrit = pymgrit.Mgrit(problem=problem, tol=1e-10)
    sol = mgrit.solve()
