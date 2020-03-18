*************
Heat Equation
*************

-----------------
1-D heat equation
-----------------

* source: heat_1d.py_ (basic) or vector_heat_1d_2pts.py_, heat_1d_2pts_bdf1.py_, heat_1d_2pts_bdf2.py_ (advanced)

* example codes: example_heat_1d.py_, example_heat_1d_bdf2.py_

* scalar PDE with unknown function :math:`u(x,t)` of two independent variables

* discretization:

  * second-order central finite differences in space
  * time integration:

    * backward Euler
    * BDF2

.. _example_heat_1d.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_1d.py
.. _example_heat_1d_bdf2.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_1d_bdf2.py

.. _heat_1d.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/heat/heat_1d.py
.. _heat_1d_2pts_bdf1.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/heat/heat_1d_2pts_bdf1.py
.. _heat_1d_2pts_bdf2.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/heat/heat_1d_2pts_bdf2.py
.. _vector_heat_1d_2pts.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/heat/vector_heat_1d_2pts.py

The heat equation in 1D space is a partial differential equation that governs the flow of heat in a homogeneous and
isotropic medium with :math:`u(x,t)` being the temperature at the point :math:`x` at time t. Denoting the thermal
conductivity by :math:`a`, the governing equation is given by

.. math::
    u_t - au_{xx} = b(x,t) \;\; \text{ in } \; [x_{start},x_{end}]\times(t_0,t_{end}] \;\text{ with }\; u(x, t_0) = u_0(x)

and subject to some boundary conditions in space.

In example_heat_1d.py_, the heat equation in the domain :math:`[0,1]\times[0,2]` is considered with a thermal
conductivity of :math:`a = 1`, right-hand-side :math:`b(x,t)=-\sin(\pi x) (\sin(t) - \pi^2 \cos(t))`, homogeneous
Dirichlet boundary conditions in space, and subject to the initial condition :math:`u(x,0) = \sin(\pi x)`.

Look at example_heat_1d_bdf2.py_ for using a combination of BDF2 and backward Euler time integration.


-----------------
2-D heat equation
-----------------

* source: heat_2d.py_

* example code: example_heat_2d.py_

* scalar PDE with unknown function :math:`u(x, y, t)` of three independent variables

* discretization:

  * second-order central finite differences in space
  * time integration:

    * backward Euler
    * forward Euler
    * Crank-Nicolson

.. _example_heat_2d.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_heat_2d.py

.. _heat_2d.py: https://github.com/pymgrit/pymgrit/tree/master/src/pymgrit/heat/heat_2d.py

The heat equation in 2D space is a partial differential equation that governs the flow of heat in a homogeneous and
isotropic medium with :math:`u(x, y, t)` being the temperature at the point :math:`(x,y)` at time :math:`t`.
Denoting the thermal conductivity by :math:`a`, the governing equation is given by

.. math::
    u_t - a(u_{xx}+u_{yy}) = b(x,y,t) \;\; \text{ in } \; [0,x_{end}]\times[0,y_{end}]\times(t_0,t_{end}] \;\text{ with }\;
    u(x,y, t_0) = u_0(x,y)

and subject to some boundary conditions in space.

In example_heat_2d.py_, the heat equation in the domain :math:`[0,0.75]\times[0,1.5]\times[0,2]` is considered with a
thermal conductivity of :math:`a = 3.5`, right-hand-side
:math:`b(x,t) = 5x(x_{end}-x)y(y_{end}-y) + 10at(y(y_{end}-y) + x(x_{end} - x)`,
homogeneous Dirichlet boundary conditions in space, and subject to the initial condition :math:`u(x,y,0) = 0`.

