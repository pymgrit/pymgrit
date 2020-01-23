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

TODO

PyMGRIT Features
----------------

TODO

Example Usage
----------------

PyMGRIT is easy to use! The following code constructs a 1-d heat equation example and solves the resulting linear system
with the MGRIT algorithm::

    import pymgrit
    heat_lvl_0 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65)
    heat_lvl_1 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17)
    problem = [heat_lvl_0, heat_lvl_1]
    transfer = [pymgrit.GridTransferCopy()]
    mgrit = pymgrit.Mgrit(problem=problem, transfer=transfer, tol=1e-10)
    sol = mgrit.solve()

Program output::

    INFO - 23-01-20 11:00:25 - Start setup
    INFO - 23-01-20 11:00:25 - Setup took 0.12172603607177734 s
    INFO - 23-01-20 11:00:31 - Start solve
    INFO - 23-01-20 11:00:31 - iter 1  | con: 0.010493952134079353   | con-fac: -                       | runtime: 0.24430084228515625 s
    INFO - 23-01-20 11:00:32 - iter 2  | con: 0.0006626496424129124  | con-fac: 0.06314586096318682     | runtime: 0.10121035575866699 s
    INFO - 23-01-20 11:00:32 - iter 3  | con: 3.9736223108553456e-05 | con-fac: 0.059965659928316824    | runtime: 0.08508825302124023 s
    INFO - 23-01-20 11:00:32 - iter 4  | con: 2.087662565787507e-06  | con-fac: 0.052538022048152964    | runtime: 0.07944488525390625 s
    INFO - 23-01-20 11:00:32 - iter 5  | con: 8.482152793325008e-08  | con-fac: 0.04062990318612805     | runtime: 0.10628175735473633 s
    INFO - 23-01-20 11:00:32 - iter 6  | con: 2.120708067856101e-09  | con-fac: 0.025002002669946982    | runtime: 0.0875704288482666 s
    INFO - 23-01-20 11:00:32 - iter 7  | con: 2.003907620345022e-11  | con-fac: 0.009449238444077046    | runtime: 0.09730362892150879 s
    INFO - 23-01-20 11:00:32 - Solve took 0.8652820587158203 s
    INFO - 23-01-20 11:00:32 - Run parameter overview

    interval                  : [0.0, 2.0]
    number points             : 65 points
    max dt                    : 0.03125
    level                     : 2
    coarsening                : [4]
    cf_iter                   : 1
    nested iteration          : True
    cycle type                : V
    stopping tolerance        : 1e-10
    communicator size time    : 1
    communicator size space   : -99


