|Test Status| |Lint Status| |Docs Status| |Publish Status|

.. |Lint Status| image:: https://github.com/pymgrit/pymgrit/workflows/Lint/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ALint
.. |Test Status| image:: https://github.com/pymgrit/pymgrit/workflows/Test/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ATest
.. |Docs Status| image:: https://github.com/pymgrit/pymgrit/workflows/Docs/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ADocs
.. |Publish Status| image:: https://github.com/pymgrit/pymgrit/workflows/Publish/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3APublish

Introduction
------------

PyMGRIT is a package for the Multigrid-Reduction-in-Time (MGRIT) algorithm in Python.

PyMGRIT is currently developed by `Jens Hahne`_ and `Stephanie Friedhoff`_.

.. _Jens Hahne: https://www.hpc.uni-wuppertal.de/en/scientific-computing-and-high-performance-computing/members/jens-hahne.html

.. _Stephanie Friedhoff: https://www.hpc.uni-wuppertal.de/en/scientific-computing-and-high-performance-computing/members/dr-stephanie-friedhoff.html


What is MGRIT?
---------------

The MGRIT algorithm is a reduction-based time-multigrid method for solving time-dependent problems. In contrast to
solving sequentially for one time step after the other, the MGRIT algorithm is an iterative method that allows
calculating multiple time steps simultaneously by using a time-grid hierarchy. The MGRIT method is a non-intrusive
approach that essentially uses the same time integrator as a traditional time-stepping algorithm. Therefore, it is
particularly well suited for introducing time parallelism in simulations using existing application codes.

PyMGRIT Features
----------------

PyMGRIT features:

* Classical Multigrid-Reduction-in-Time (MGRIT) for solving evolutionary systems of equations

  * Non-intrusive approach
  * Optimal time-multigrid algorithm
  * A variety of cycling strategies, relaxation schemes, and coarsening strategies

* Time parallelism

* Specific to space-time problems

  * Space & time parallelism
  * Additional coarsening in space

Citing
------

::

    @MISC{PyMGRIT,
      author = "Hahne, J. and Friedhoff, S.",
      title = "{PyMGRIT}: Multigrid-Reduction-in-Time in {Python} v1.0",
      year = "2020",
      url = "https://github.com/pymgrit/pymgrit",
      note = "Release 1.0"
      }

Installation
------------

PyMGRIT requires ``mpicc`` (from ``openmpi`` or ``mpich``)

    >>> pip3 install pymgrit

or

    >>> git clone https://github.com/pymgrit/pymgrit.git
    >>> cd pymgrit
    >>> pip3 install .

Example Usage
----------------

PyMGRIT is easy to use! The following code generates a discrete Dahlquist test problem and solves the resulting linear
system using a two-level MGRIT algorithm.::

    # Import PyMGRIT
    from pymgrit import *

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the solver tolerance to 1e-10
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

    # Solve the test problem
    info = mgrit.solve()

Program output::

    INFO - 21-02-20 16:18:43 - Start setup
    INFO - 21-02-20 16:18:43 - Setup took 0.009232759475708008 s
    INFO - 21-02-20 16:18:43 - Start solve
    INFO - 21-02-20 16:18:43 - iter 1  | conv: 7.186185937031941e-05  | conv factor: -                       | runtime: 0.013237237930297852 s
    INFO - 21-02-20 16:18:43 - iter 2  | conv: 1.2461067076355103e-06 | conv factor: 0.017340307063501627    | runtime: 0.010195493698120117 s
    INFO - 21-02-20 16:18:43 - iter 3  | conv: 2.1015566145245807e-08 | conv factor: 0.016864981158092696    | runtime: 0.008922338485717773 s
    INFO - 21-02-20 16:18:43 - iter 4  | conv: 3.144127445017594e-10  | conv factor: 0.014960945726074891    | runtime: 0.0062139034271240234 s
    INFO - 21-02-20 16:18:43 - iter 5  | conv: 3.975214076032893e-12  | conv factor: 0.01264329816633959     | runtime: 0.006150722503662109 s
    INFO - 21-02-20 16:18:43 - Solve took 0.05394101142883301 s
    INFO - 21-02-20 16:18:43 - Run parameter overview
      time interval             : [0.0, 5.0]
      number of time points     : 101
      max dt                    : 0.05000000000000071
      number of levels          : 2
      coarsening factors        : [2]
      cf_iter                   : 1
      nested iteration          : True
      cycle type                : V
      stopping tolerance        : 1e-10
      time communicator size    : 1
      space communicator size   : -99


Getting Help
------------

For documentation see https://pymgrit.github.io/pymgrit/

Create an issue_.

.. _issue: https://github.com/pymgrit/pymgrit/issues

Look at the Quickstart_, Tutorial_ or the Examples_.

.. _Examples: https://pymgrit.github.io/pymgrit/usage/examples.html
.. _Tutorial: https://pymgrit.github.io/pymgrit/usage/tutorial.html
.. _Quickstart: https://pymgrit.github.io/pymgrit/usage/quickstart.html

