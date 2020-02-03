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
    - `clone_zero`: Initialization of the data with zeros
    - `clone_rand`: Initialization of the data with random values
    - `set_values`: Sets the solution
    - `get_values`: Gets the solution

::

    import numpy as np
    from pymgrit.core.vector import Vector

    class VectorDahlquist(Vector):
        """
        Vector for the dahlquist test equation
        """

        def __init__(self, value):
            super(VectorDahlquist, self).__init__()
            self.value = value

        def __add__(self, other):
            tmp = VectorDahlquist(0)
            tmp.set_values(self.get_values() + other.get_values())
            return tmp

        def __sub__(self, other):
            tmp = VectorDahlquist(0)
            tmp.set_values(self.get_values() - other.get_values())
            return tmp

        def norm(self):
            return np.linalg.norm(self.value)

        def clone_zero(self):
            return VectorDahlquist(0)

        def clone_rand(self):
            tmp = VectorDahlquist(0)
            tmp.set_values(np.random.rand(1)[0])
            return tmp

        def set_values(self, value):
            self.value = value

        def get_values(self):
            return self.value

-----------
Application
-----------

TODO!

The next step is to write your first problem application. Therefore, write a class that inherits from `Application`.
The variables `vector_template` and 'vector_t_start' have to be specified and set
`step` has to be specified::

    from pymgrit.core.application import Application

    class Dahlquist(Application):
        """
        Solves  u' = lambda u,
        with lambda=-1 and y(0) = 1
        """

        def __init__(self, *args, **kwargs):
            super(Dahlquist, self).__init__(*args, **kwargs)
            self.vector_template = VectorDahlquist(0)  # Setting the class which is used for each time point
            self.vector_t_start = VectorDahlquist(1)  # Setting the initial condition

        def step(self, u_start: VectorDahlquist, t_start: float, t_stop: float) -> VectorDahlquist:
            tmp = 1 / (1 + t_stop - t_start) * u_start.get_values()
            ret = VectorDahlquist(tmp)
            return ret

-----------------
Solve the problem
-----------------

TODO::

    from pymgrit import *

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_strucutre = simple_setup_problem(problem=dahlquist, level=2,coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_strucutre, tol = 1e-10)

    # Solve
    mgrit.solve()
