Welcome to PyMGRIT's documentation!
===================================

PyMGRIT is a package for the Multigrid-Reduction-in-Time (MGRIT) algorithm in Python.

-----
MGRIT
-----

The MGRIT algorithm is a reduction-based time-multigrid method for solving time-dependent problems. In contrast to
solving sequentially for one time step after the other, the MGRIT algorithm is an iterative method that allows
calculating multiple time steps simultaneously by using a time-grid hierarchy. The MGRIT method is a non-intrusive
approach that essentially uses the same time integrator as a traditional time-stepping algorithm. Therefore, it is
particularly well suited for introducing time parallelism in simulations using existing application codes.

--------
Overview
--------

* Get PyMGRIT :doc:`installed <usage/installation>`.
* :doc:`usage/quickstart`
* :doc:`Implement <usage/tutorial>` a simple problem in PyMGRIT.
* Learn about :doc:`basic <usage/examples>` and :doc:`advanced <usage/advanced>` PyMGRIT features.
* Run :doc:`parallel simulations <usage/parallelism>`.


------------
Getting help
------------

* Try the :doc:`FAQ <help/faq>`.
* Looking for specific information? Try the :ref:`genindex` or :ref:`modindex`.
* Report bugs with PyMGRIT in our `issue tracker`_.

.. _issue tracker: https://github.com/pymgrit/pymgrit/issues

------
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

.. toctree::
    :caption: Usage
    :hidden:

    usage/installation
    usage/quickstart
    usage/tutorial
    usage/examples
    usage/parallelism
    usage/advanced

.. toctree::
    :caption: Applications
    :hidden:

    applications/dahlquist
    applications/brusselator
    applications/arenstorf_orbit
    applications/heat_equation
    applications/advection


.. toctree::
    :caption: Coupling
    :hidden:

    coupling/firedrake

.. toctree::
    :caption: Help
    :hidden:

    help/faq
