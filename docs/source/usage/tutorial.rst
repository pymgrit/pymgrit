**********
Tutorial
**********

This tutorial demonstrates basic usage of the PyMGRIT package. Our goal is solving Dahlquist's test problem,

.. math::
    u' = \lambda u \;\;\text{ in } (0, 5] \text{ with }\; u(0) = 1,

discretized by Backward Euler. To accomplish this, this tutorial will go through the following tasks:

#. Writing the `vector structure`_ for all time-dependent information
#. Writing the `application`_ class holding any time-independent data
#. `Solving the problem`_

-----------------
Vector structure
-----------------

The first step is to write a data structure that contains the solution of a single point in time. The data structure must inherit from PyMGRIT's core `Vector` class.

For our test problem, we import PyMGRIT's core `Vector` class (and numpy for later use)::

    import numpy as np
    from pymgrit.core.vector import Vector

Then, we define the class `VectorDahlquist` containing a scalar member variable `value`::

    class VectorDahlquist(Vector):
        """
        Vector for Dahlquist's test equation
        """

        def __init__(self, value):
            super(VectorDahlquist, self).__init__()
            self.value = value

Furthermore, we must define the following seven member functions: , `set_values`, `get_values`, `clone_zero`, `clone_rand`, `__add__`, `__sub__`, and `norm`.

The function `set_values` receives data values and overwrites the values of the vector data and `get_values` returns the vector data. For our class `VectorDahlquist`, we define the following member functions::

        def set_values(self, value):
            self.value = value

        def get_values(self):
            return self.value

The function `clone_zero` returns a vector object initialized with zeros; `clone_rand` similarly returns a vector object initialized with random data. For our class `VectorDahlquist`, these member functions are defined as follows::

        def clone_zero(self):
            return VectorDahlquist(0)

        def clone_rand(self):
            tmp = VectorDahlquist(0)
            tmp.set_values(np.random.rand(1)[0])
            return tmp

The functions `__add__`, `__sub__`, and `norm` define the addition and subtraction of two vector objects or the norm of a vector object, respectively. For our class `VectorDahlquist`, we define these member functions as follows::

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

Summary
^^^^^^^
The vector class must inherit from PyMGRIT's core `Vector` class.

Member variables hold all data of a single time point.

The following member functions must be defined:

    - `__add__`: Addition of two vector objects
    - `__sub__`: Subtraction of two vector objects
    - `norm`: Norm of a vector object (for measuring convergence)
    - `clone_zero`: Initialization of vector data with zeros
    - `clone_rand`: Initialization of vector data with random values
    - `set_values`: Setting vector data
    - `get_values`: Getting vector data

::

    import numpy as np
    from pymgrit.core.vector import Vector

    class VectorDahlquist(Vector):
        """
        Vector for Dahlquist's test equation
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
Solving the problem
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
