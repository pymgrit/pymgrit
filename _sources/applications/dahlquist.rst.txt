*********
Dahlquist
*********

* source: dahlquist.py_

* example code: example_dahlquist.py_

* scalar ODE

* discretization: backward Euler

.. _example_dahlquist.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_dahlquist.py

.. _dahlquist.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/dahlquist/dahlquist.py

The simplest scalar ODE, made famous by Germund Dahlquist, is given by

.. math::
    u' = \lambda u \;\;\text{ in } (t_0, t_{end}]\;\; \text{ with }\; u(t_0) = 1,

and constant :math:`\lambda < 0`.
The exact solution :math:`u(t) = e^{\lambda(t-t_0)}u(t_0)` decays to zero as time increases.

For :math:`t_0 = 0, t_{end} = 5,` and :math:`\lambda = -1`, for example, we obtain the following solution:

.. figure:: ../figures/dahlquist.png
    :alt: dahlquist solution



