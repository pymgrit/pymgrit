|Test Status| |Lint Status|

.. |Test Status| image:: https://github.com/pymgrit/pymgrit/workflows/Lint/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions
.. |Lint Status| image:: https://github.com/pymgrit/pymgrit/workflows/Test/badge.svg
   :target: https://github.com/pymgrit/pymgrit/actions

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

For documentation see (TODO)

Create an issue (TODO)

Look at the Tutorial(TODO) or the Examples(TODO)

What is MGRIT?
---------------

TODO

PyMGRIT Features
----------------

TODO

Example Usage
----------------

    >>> import pymgrit
    >>> heat_lvl_0 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=65)
    >>> heat_lvl_1 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1, t_start=0, t_stop=2, nt=17)
    >>> problem = [heat_lvl_0, heat_lvl_1]
    >>> transfer = [pymgrit.GridTransferCopy()]
    >>> mgrit = pymgrit.Mgrit(problem=problem, transfer=transfer, tol=1e-10)
    >>> sol = mgrit.solve()
