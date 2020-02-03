**********
Quickstart
**********

PyMGRIT is easy to use! The following code generates an example of dahlquist test equation u' = lambda u and solves the resulting problem with the MGRIT algorithm.::

    from pymgrit import *

    # Creating the finest level problem
    dahlquist = Dahlquist(t_start=0, t_stop=5, nt=101)

    # Setup the multilevel structure by using the simple_setup_problem function
    dahlquist_multilevel_strucutre = simple_setup_problem(problem=dahlquist, level=2, coarsening=2)

    # Setup of the MGRIT algorithm with the multilevel structure
    mgrit = Mgrit(problem=dahlquist_multilevel_strucutre, tol=1e-10)

    # Solve
    return mgrit.solve()

Program output::

    INFO - 03-02-20 11:19:03 - Start setup
    INFO - 03-02-20 11:19:03 - Setup took 0.009920358657836914 s
    INFO - 03-02-20 11:19:03 - Start solve
    INFO - 03-02-20 11:19:03 - iter 1  | con: 7.186185937031941e-05  | con-fac: -                       | runtime: 0.01379704475402832 s
    INFO - 03-02-20 11:19:03 - iter 2  | con: 1.2461067076355103e-06 | con-fac: 0.017340307063501627    | runtime: 0.007235527038574219 s
    INFO - 03-02-20 11:19:03 - iter 3  | con: 2.1015566145245807e-08 | con-fac: 0.016864981158092696    | runtime: 0.005523681640625 s
    INFO - 03-02-20 11:19:03 - iter 4  | con: 3.144127445017594e-10  | con-fac: 0.014960945726074891    | runtime: 0.004599332809448242 s
    INFO - 03-02-20 11:19:03 - iter 5  | con: 3.975214076032893e-12  | con-fac: 0.01264329816633959     | runtime: 0.0043201446533203125 s
    INFO - 03-02-20 11:19:03 - Solve took 0.042092084884643555 s
    INFO - 03-02-20 11:19:03 - Run parameter overview
    interval                  : [0.0, 5.0]
    number points             : 101 points
    max dt                    : 0.05000000000000071
    level                     : 2
    coarsening                : [2]
    cf_iter                   : 1
    nested iteration          : True
    cycle type                : V
    stopping tolerance        : 1e-10
    communicator size time    : 1
    communicator size space   : -99
