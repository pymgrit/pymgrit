**********
Quickstart
**********

PyMGRIT is easy to use! In the following, we generate a discrete Dahlquist test problem and solve the resulting linear system using a two-level MGRIT algorithm.

Look at the Application :doc:`Dahlquist <../applications/dahlquist>` for more information about this test problem.

First, import PyMGRIT::

    from pymgrit import *

Create Dahlquist's test problem for the time interval [0, 5] with 101 equidistant time points (100 time points + 1 time point for the initial time t = 0)::

    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

Construct a multigrid hierarchy for the test problem `dahlquist`::

    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

This tells PyMGRIT to set up a hierarchy with two temporal grid levels using the test problem `dahlquist` and a temporal coarsening factor of two, i.e., on the fine grid, the number of time points is 101, and on the coarse grid, 51 (=100/2+1) time points are used.

Set up the MGRIT solver for the test problem using `dahlquist_multilevel_structure` and set the solver tolerance to 1e-10::

    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

which gives::

    INFO - 21-02-20 16:18:43 - Start setup
    INFO - 21-02-20 16:18:43 - Setup took 0.009232759475708008 s

Finally, solve the test problem using the `solve()` routine of the solver `mgrit`::

    mgrit.solve()

producing the output::

    INFO - 21-02-20 16:18:43 - Start solve
    INFO - 21-02-20 16:18:43 - iter 1  | con: 7.186185937031941e-05  | con-fac: -                       | runtime: 0.013237237930297852 s
    INFO - 21-02-20 16:18:43 - iter 2  | con: 1.2461067076355103e-06 | con-fac: 0.017340307063501627    | runtime: 0.010195493698120117 s
    INFO - 21-02-20 16:18:43 - iter 3  | con: 2.1015566145245807e-08 | con-fac: 0.016864981158092696    | runtime: 0.008922338485717773 s
    INFO - 21-02-20 16:18:43 - iter 4  | con: 3.144127445017594e-10  | con-fac: 0.014960945726074891    | runtime: 0.0062139034271240234 s
    INFO - 21-02-20 16:18:43 - iter 5  | con: 3.975214076032893e-12  | con-fac: 0.01264329816633959     | runtime: 0.006150722503662109 s
    INFO - 21-02-20 16:18:43 - Solve took 0.05394101142883301 s
    INFO - 21-02-20 16:18:43 - Run parameter overview
    time interval             : [0.0, 5.0]
    number of time points     : 101 points
    max dt                    : 0.05000000000000071
    number of levels          : 2
    coarsening factors        : [2]
    cf_iter                   : 1
    nested iteration          : True
    cycle type                : V
    stopping tolerance        : 1e-10
    time communicator size    : 1
    space communicator size   : -99


Summary
-------
The following code generates a discrete Dahlquist test problem and solves the resulting linear system using the MGRIT algorithm with two time levels::

    # Import PyMGRIT
    from pymgrit import *

    # Create Dahlquist's test problem with 101 time steps in the interval [0, 5]
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Construct a two-level multigrid hierarchy for the test problem using a coarsening factor of 2
    dahlquist_multilevel_structure = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Set up the MGRIT solver for the test problem and set the solver tolerance to 1e-10
    mgrit = Mgrit(problem=dahlquist_multilevel_structure, tol=1e-10)

    # Solve the test problem
    mgrit.solve()