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
    >>> heat_lvl_0 = pymgrit.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                       t_start=0, t_stop=2, nt=65)
    >>> heat_lvl_1 = pymgrit.heat_equation.HeatEquation(x_start=0, x_end=2, nx=1001, d=1,
                                       t_start=0, t_stop=2, nt=17)
    >>> problem = [heat0, heat1]
    >>> transfer = [pymgrit.core.grid_transfer_copy.GridTransferCopy()]
    >>> mgrit = solver.Mgrit(problem=problem, transfer=transfer, tol=1e-10)
    >>> mgrit.solve()
