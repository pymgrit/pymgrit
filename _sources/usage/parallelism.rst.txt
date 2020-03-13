***********
Parallelism
***********

This page contains short examples that demonstrate the parallel usage of PyMGRIT.

    - `Time parallelism`_
    - `Space & time parallelism`_

----------------
Time parallelism
----------------

Use `mpirun` to start an example in parallel::

    mpirun -np 2 python3 example_dahlquist.py

Output function
^^^^^^^^^^^^^^^

example_output_fcn.py_

.. _example_output_fcn.py: https://github.com/pymgrit/pymgrit/tree/master/examples/example_output_fcn.py

In this example, we show how to use the output function for a parallel run and save the solution after each iteration.
Each process calls the function, so the file of each process must be unique. We use the rank to distinguish the files::

    def output_fcn(self):
        # Set path to solution; here, we include the iteration number in the path name
        path = 'results/' + 'brusselator' + '/' + str(self.solve_iter)
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # Save solution to file.
        # Useful member variables of MGRIT solver:
        #   - self.t[0]           : local fine-grid (level 0) time interval
        #   - self.index_local[0] : indices of local fine-grid (level 0) time interval
        #   - self.u[0]           : fine-grid (level 0) solution values
        #   - self.comm_time_rank : Time communicator rank
        np.save(path + '/brusselator-' + str(self.comm_time_rank),
                [self.u[0][i].get_values() for i in self.index_local[0]])

Then, we solve the problem in the usual way::

    # Create two-level time-grid hierarchy for the Brusselator system
    brusselator_lvl_0 = Brusselator(t_start=0, t_stop=12, nt=641)
    brusselator_lvl_1 = Brusselator(t_interval=brusselator_lvl_0.t[::20])

    # Set up the MGRIT solver using the two-level hierarchy and set the output function
    mgrit = Mgrit(problem=[brusselator_lvl_0, brusselator_lvl_1], output_fcn=output_fcn, output_lvl=2, cf_iter=0)

    # Solve Brusselator system
    info = mgrit.solve()

The last step is to plot the solution for each iteration. Therefore, all files have to be loaded and the solution has to
be assembled in the correct order. To determine the correct order we use the rank that is part of the file name::

    if MPI.COMM_WORLD.Get_rank() == 0:
        iterations_needed = len(info['conv']) + 1
        cols = 2
        rows = iterations_needed // cols + iterations_needed % cols
        position = range(1, iterations_needed + 1)
        fig = plt.figure(1, figsize=[10, 10])
        for i in range(iterations_needed):
            files = []
            path = 'results/brusselator/' + str(i) + '/'
            # Construct solution from multiple files
            for filename in os.listdir(path):
                files.append([int(filename[filename.find('-') + 1: -4]), np.load(path + filename, allow_pickle=True)])
            files.sort(key=lambda tup: tup[0])
            sol = np.vstack([l.pop(1) for l in files])
            ax = fig.add_subplot(rows, cols, position[i])
            # Plot the two solution values at each time point
            ax.scatter(sol[:, 0], sol[:, 1])
            ax.set(xlabel='x', ylabel='y')
        fig.tight_layout(pad=2.0)
        plt.show()


Scaling results
^^^^^^^^^^^^^^^

------------------------
Space & time parallelism
------------------------

To be done.