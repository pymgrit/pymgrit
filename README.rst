|Test Status| |Lint Status| |Docs Status| |Publish Status|

.. |Lint Status| image:: https://github.com/pymgrit/pymgrit/workflows/Lint/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ALint
.. |Test Status| image:: https://github.com/pymgrit/pymgrit/workflows/Test/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ATest
.. |Docs Status| image:: https://github.com/pymgrit/pymgrit/workflows/Docs/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3ADocs
.. |Publish Status| image:: https://github.com/pymgrit/pymgrit/workflows/Publish/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions?query=workflow%3APublish

Installation
------------

PyMGRIT requires ``mpicc`` (from ``openmpi`` or ``mpich``)

    >>> pip install pymgrit

or

    >>> python setup.py install

Introduction
------------

PyMGRIT is a library for the Multigrid-Reduction-in-Time (MGRIT) algorithm in Python

Getting Help
------------

For documentation see https://pymgrit.github.io/pymgrit/

Create an issue_.

.. _issue: https://github.com/pymgrit/pymgrit/issues

Look at the Quickstart_ or the Examples_.

.. _Examples: https://pymgrit.github.io/pymgrit/usage/examples.html
.. _Quickstart: https://pymgrit.github.io/pymgrit/usage/quickstart.html

What is MGRIT?
---------------

To be done.

PyMGRIT Features
----------------

    - Classical Multigrid-Reduction-in-Time (MGRIT)
    - Additional coarsening in space

Example Usage
----------------

PyMGRIT is easy to use! The following code constructs a 1-d heat equation example and solves the resulting space-time
system of equations with MGRIT::

    import pymgrit
    heat_lvl_0 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65)
    heat_lvl_1 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17)
    heat_lvl_2 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=5)
    problem = [heat_lvl_0, heat_lvl_1, heat_lvl_2]
    mgrit = pymgrit.Mgrit(problem=problem, tol=1e-10)
    sol = mgrit.solve()

Program output::

    INFO - 23-01-20 11:48:53 - Start setup
    INFO - 23-01-20 11:48:53 - Setup took 0.1085507869720459 s
    INFO - 23-01-20 11:48:54 - Start solve
    INFO - 23-01-20 11:48:54 - iter 1  | con: 0.010531208413419799   | con-fac: -                       | runtime: 0.21976184844970703 s
    INFO - 23-01-20 11:48:55 - iter 2  | con: 0.0006683816775940518  | con-fac: 0.06346676006737657     | runtime: 0.15288186073303223 s
    INFO - 23-01-20 11:48:55 - iter 3  | con: 4.0050232837502924e-05 | con-fac: 0.05992120098454857     | runtime: 0.12258291244506836 s
    INFO - 23-01-20 11:48:55 - iter 4  | con: 2.0920119846420314e-06 | con-fac: 0.052234702183381       | runtime: 0.13314509391784668 s
    INFO - 23-01-20 11:48:55 - iter 5  | con: 8.482152793325008e-08  | con-fac: 0.040545431171496886    | runtime: 0.13439655303955078 s
    INFO - 23-01-20 11:48:55 - iter 6  | con: 2.120708067856101e-09  | con-fac: 0.025002002669946982    | runtime: 0.12366461753845215 s
    INFO - 23-01-20 11:48:55 - iter 7  | con: 2.003907620345022e-11  | con-fac: 0.009449238444077046    | runtime: 0.15373992919921875 s
    INFO - 23-01-20 11:48:55 - Solve took 1.1199557781219482 s
    INFO - 23-01-20 11:48:55 - Run parameter overview

    interval                  : [0.0, 2.0]
    number points             : 65 points
    max dt                    : 0.03125
    level                     : 3
    coarsening                : [4, 4]
    cf_iter                   : 1
    nested iteration          : True
    cycle type                : V
    stopping tolerance        : 1e-10
    communicator size time    : 1
    communicator size space   : -99



