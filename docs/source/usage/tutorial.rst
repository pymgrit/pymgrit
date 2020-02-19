**********
Tutorial
**********

In the following tutorial you will write your first own application and use the MGRIT algorithm to solve the time dependent problem. To accomplish this, this tutorial will go through the following tasks:

#. Write your first `vector structure`_
#. Write your first `application`_
#. `Solve the problem`_

-----------------
Vector structure
-----------------

The first step is to write a data structure that contains the solution of a point in time. The data structure must inherit from the 'Vector' class, which is at the core of PyMGRIT. Our class receives a size as an integer and then generates a numpy array of that size containing the solutions. Furthermore the following functions must be defined:

    - `__add__`: For the addition of two objects of our class
    - `__sub__`: For the substraction of two objects of our class
    - `norm`: Some norm for measuring the convergence
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

In the next step we write our first application. In our case we use Dahlquist test equation. Every application must inherit from the class 'Application' from the PyMGRIT core. The superclass takes care of generating the time interval. Our application must contain the following information:

    - Variable: `vector_template` : Selected as data structure for each event
    - Variable: `vector_t_start` : Same data structure. Set the initial conditions here
    - Function: `step` : Time integrator

::

    #Import superclass Application
    from pymgrit.core.application import Application

    class Dahlquist(Application):
        """
        Solves  u' = lambda u,
        with lambda=-1 and y(0) = 1
        """

        def __init__(self, *args, **kwargs):
            super(Dahlquist, self).__init__(*args, **kwargs)

            # Setting the class which is used for each time point (mandatory)
            self.vector_template = VectorDahlquist(0)

            # Setting the initial condition (mandatory)
            self.vector_t_start = VectorDahlquist(1)

        def step(self, u_start: VectorDahlquist, t_start: float, t_stop: float) -> VectorDahlquist:
            tmp = 1 / (1 + t_stop - t_start) * u_start.get_values()
            return VectorDahlquist(tmp)

-----------------
Solve the problem
-----------------

The last step is to create an object of the application. Using the application object and the function 'simple_setup_problem' from the PyMGRIT core a multilevel structure is created. This is passed to the MGRIT algorithm and solved.::

    from pymgrit import *

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2,coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol = 1e-10)

    # Solve
    mgrit.solve()
