"""
Example demonstrating the coupling with petsc4py.

Apply three-level MGRIT V-cycles with FCF-relaxation to solve a 2D heat equation problem.

Note: This example requires petsc4py!
"""

import pathlib
import os
import numpy as np

from mpi4py import MPI

from pymgrit.core.split import split_communicator
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.grid_transfer_copy import GridTransferCopy
from pymgrit.petsc.heat_2D_petsc import HeatPetsc
from pymgrit.petsc.heat_2D_petsc import GridTransferPetsc

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


def main():
    def output_fcn(self):
        # Set path to solution
        path = 'results/petsc'
        # Create path if not existing
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save solution with corresponding time point to file
        np.save(path + '/petsc' + str(self.comm_time_rank) + str(self.comm_space_rank),
                [[[self.t[0][i], self.comm_space_rank, self.u[0][i].get_values().getArray()] for i in
                  self.index_local[0]]])

    # Split the communicator into space and time communicator
    comm_world = MPI.COMM_WORLD
    comm_x, comm_t = split_communicator(comm_world, 4)

    # Create PETSc DMDA grids
    nx = 129
    ny = 129
    dmda_coarse = PETSc.DMDA().create([nx, ny], stencil_width=1, comm=comm_x)
    dmda_fine = dmda_coarse.refine()

    # Set up the problem
    heat_petsc_0 = HeatPetsc(dmda=dmda_fine, comm_x=comm_x, freq=1, a=1.0, t_start=0, t_stop=1, nt=33)
    heat_petsc_1 = HeatPetsc(dmda=dmda_coarse, comm_x=comm_x, freq=1, a=1.0, t_interval=heat_petsc_0.t[::2])
    heat_petsc_2 = HeatPetsc(dmda=dmda_coarse, comm_x=comm_x, freq=1, a=1.0, t_interval=heat_petsc_1.t[::2])

    # Setup three-level MGRIT solver with the space and time communicators and
    # solve the problem
    mgrit = Mgrit(problem=[heat_petsc_0, heat_petsc_1, heat_petsc_2],
                  transfer=[GridTransferPetsc(fine_prob=dmda_fine, coarse_prob=dmda_coarse), GridTransferCopy()],
                  comm_time=comm_t, comm_space=comm_x, output_fcn=output_fcn)
    info = mgrit.solve()

    import time
    if comm_t.Get_rank() == 0:
        time.sleep(1)
        sol = []
        path = 'results/petsc/'
        for filename in os.listdir(path):
            data = np.load(path + filename, allow_pickle=True).tolist()[0]
            sol += data
        sol = [item for item in sol if item[1] == comm_x.Get_rank()]
        sol.sort(key=lambda tup: tup[0])

        u_e = heat_petsc_0.u_exact(t=heat_petsc_0.t[-1]).get_values().getArray()
        diff = sol[-1][2] - u_e
        print('Difference at time point', heat_petsc_0.t[-1], ':',
              np.linalg.norm(diff, np.inf), '(space rank', comm_x.Get_rank(), ')')


if __name__ == '__main__':
    main()
