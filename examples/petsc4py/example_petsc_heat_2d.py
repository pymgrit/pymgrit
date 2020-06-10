"""
Example demonstrating the coupling with petsc4py.

Apply three-level MGRIT V-cycles with FCF-relaxation to solve a 2D heat equation problem.

Note: This example requires petsc4py!
"""

import pathlib
import os
import copy
import numpy as np

from mpi4py import MPI

from pymgrit.core.split import split_communicator
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.grid_transfer import GridTransfer
from pymgrit.core.grid_transfer_copy import GridTransferCopy
from pymgrit.core.vector import Vector as PymgritVector
from pymgrit.core.application import Application

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This examples requires petsc4py.")


class VectorPetsc(PymgritVector):
    """
    Vector class for PETSc vectors
    """

    def __init__(self, values: PETSc.Vec) -> None:
        """
        Constructor.

        :param values: PETSc.Vec with approximation
        """
        if isinstance(values, PETSc.Vec):
            self.values = copy.deepcopy(values)
        else:
            raise Exception('Wrong datatype')

    def __add__(self, other: '__class__') -> '__class__':
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        return VectorPetsc(self.get_values() + other.get_values())

    def __sub__(self, other: '__class__') -> '__class__':
        """
        Subtraction of two vector objects (self and other)

        :param other: vector object to be subtracted from self
        :return: difference of vector object self and input object other
        """
        return VectorPetsc(self.get_values() - other.get_values())

    def norm(self) -> float:
        """
        Norm of a vector object

        :return: Frobenius-norm of vector object
        """
        return self.values.norm(PETSc.NormType.FROBENIUS)

    def clone(self) -> '__class__':
        """
        Initialize vector object with copied values

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values())

    def clone_zero(self) -> '__class__':
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values() * 0)

    def clone_rand(self) -> '__class__':
        """
        Initialize vector object with random values

        :rtype: vector object with random values
        """
        tmp = VectorPetsc(self.get_values())
        return tmp

    def set_values(self, values: PETSc.Vec) -> None:
        """
        Set vector data

        :param values: values for vector object
        """
        self.values = values

    def get_values(self) -> PETSc.Vec:
        """
        Get vector data

        :return: values of vector object
        """
        return self.values

    def pack(self) -> np.ndarray:
        """
        Pack data

        :return: values of vector object
        """
        return self.values.getArray()

    def unpack(self, values: np.ndarray) -> None:
        """
        Unpack and set data

        :param values: values for vector object
        """
        self.values.setArray(values)


class HeatPetsc(Application):
    """
    2D heat equation application with Dirichlet BCs in [0,1]x[0,1]
    """

    def __init__(self, dmda: PETSc.DMDA, comm_x: MPI.Comm, freq: int, a: float, rtol: float = 1e-10,
                 atol: float = 1e-10, max_it: int = 100, *args, **kwargs) -> None:
        """
        Constructor
        :param dmda: PETSc DMDA grid
        :param comm_x: space communicator
        :param freq: frequency
        :param a: diffusion coefficient
        :param rtol: spatial solver tolerance
        :param atol: spatial solver tolerance
        :param max_it: spatial solver max iter
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.dmda = dmda
        self.mx, self.my = self.dmda.sizes
        self.dx = 1.0 / (self.mx - 1)
        self.dy = 1.0 / (self.my - 1)
        (self.xs, self.xe), (self.ys, self.ye) = self.dmda.ranges
        self.freq = freq
        self.a = a
        self.space_disc = self.get_a()
        self.id = self.get_id()

        self.vector_template = VectorPetsc(self.dmda.createGlobalVec())
        self.vector_t_start = VectorPetsc(self.u_exact(0).get_values())

        # setup solver
        self.ksp = PETSc.KSP()
        self.ksp.create(comm=comm_x)
        self.ksp.setType('gmres')
        pc = self.ksp.getPC()
        pc.setType('none')
        self.ksp.setFromOptions()
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    def step(self, u_start: VectorPetsc, t_start: float, t_stop: float) -> VectorPetsc:
        """
        Time integration routine for 2D heat equation example problem
        
        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        result = self.dmda.createGlobalVec()
        self.ksp.setOperators((t_stop - t_start) * self.space_disc + self.id)
        self.ksp.solve(self.compute_rhs(u_start=u_start, t_start=t_start, t_stop=t_stop), result)
        return VectorPetsc(result)

    def get_a(self) -> PETSc.Mat:
        """
        Define spatial discretization matrix for 2D heat equation

        Second-order central finite differences with matrix stencil
            [          -f_y          ]
            [-f_x  2(f_x + f_y)  -f_x]
            [          -f_y          ]
        with f_x = (a / dx^2) and f_y = (a / dy^2)
        :return: spatial discretization
        """
        A = self.dmda.createMatrix()
        A.setType('aij')
        A.setFromOptions()
        A.setPreallocationNNZ((5, 5))
        A.setUp()

        fx = self.a / self.dx ** 2
        fy = self.a / self.dy ** 2

        A.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        for j in range(self.ys, self.ye):
            for i in range(self.xs, self.xe):
                row.index = (i, j)
                row.field = 0
                if i == 0 or j == 0 or i == self.mx - 1 or j == self.my - 1:
                    A.setValueStencil(row, row, 0.0)
                else:
                    diag = 2 * (fx + fy)
                    for index, value in [
                        ((i, j - 1), -fx),
                        ((i - 1, j), -fy),
                        ((i, j), diag),
                        ((i + 1, j), -fy),
                        ((i, j + 1), -fx),
                    ]:
                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value)
        A.assemble()
        return A

    def get_id(self) -> PETSc.Mat:
        """
        Define identity matrix
        :return: identity matrix
        """
        Id = self.dmda.createMatrix()
        Id.setType('aij')
        Id.setFromOptions()
        Id.setPreallocationNNZ((1, 1))
        Id.setUp()

        Id.zeroEntries()
        row = PETSc.Mat.Stencil()
        (xs, xe), (ys, ye) = self.dmda.ranges
        for j in range(ys, ye):
            for i in range(xs, xe):
                row.index = (i, j)
                row.field = 0
                Id.setValueStencil(row, row, 1.0)
        Id.assemble()
        return Id

    def compute_rhs(self, u_start: VectorPetsc, t_start: float, t_stop: float) -> PETSc.Vec:
        """
        Right-hand side of spatial system

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: right-hand side of spatial system at each time step in case of implicit time integration
        """
        b = self.dmda.createGlobalVec()
        u = u_start.get_values()

        ba = self.dmda.getVecArray(b)
        ua = self.dmda.getVecArray(u)

        ba[self.xs:self.xe, self.ys:self.ye] = ua[self.xs:self.xe, self.ys:self.ye] + (
                t_stop - t_start) * self.rhs(t_stop)
        return b

    def rhs(self, t_stop: float) -> np.ndarray:
        """
        Right hand side
        :param t_stop: time point
        :return: right hand side
        """
        xv, yv = np.meshgrid(range(self.xs, self.xe), range(self.ys, self.ye), indexing='ij')
        res = -np.sin(np.pi * self.freq * xv * self.dx) * \
              np.sin(np.pi * self.freq * yv * self.dy) * \
              (np.sin(t_stop) - self.a * 2.0 * (np.pi * self.freq) ** 2 * np.cos(t_stop))
        return res

    def u_exact(self, t: float) -> VectorPetsc:
        """
        Exact solution
        :param t: time point
        :return: exact solution at point t
        """
        u = self.dmda.createGlobalVec()
        xa = self.dmda.getVecArray(u)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j] = np.sin(np.pi * self.freq * i * self.dx) * \
                           np.sin(np.pi * self.freq * j * self.dy) * np.cos(t)

        return VectorPetsc(u)


class GridTransferPetsc(GridTransfer):
    """
    Grid Transfer class between PETSc DMDA grids
    """

    def __init__(self, fine_prob: PETSc.DMDA, coarse_prob: PETSc.DMDA) -> None:
        """
        Constructor
        :param fine_prob:
        :param coarse_prob:
        """
        super().__init__()

        self.coarse_prob = coarse_prob
        self.fine_prob = fine_prob
        self.interp, _ = self.coarse_prob.createInterpolation(fine_prob)
        self.inject = self.coarse_prob.createInjection(fine_prob)

    def restriction(self, u: VectorPetsc) -> VectorPetsc:
        """
        Restriction
        :param u: fine approximation
        :return: coarse approximation
        """
        u_coarse = self.coarse_prob.createGlobalVec()
        self.inject.mult(u.get_values(), u_coarse)
        return VectorPetsc(u_coarse)

    def interpolation(self, u: VectorPetsc) -> VectorPetsc:
        """
        Interpolation
        :param u: coarse approximation
        :return: fine approximation
        """
        u_fine = self.fine_prob.createGlobalVec()
        self.interp.mult(u.get_values(), u_fine)
        return VectorPetsc(u_fine)


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
              np.linalg.norm(diff, np.inf), '(space rank',comm_x.Get_rank() , ')')


if __name__ == '__main__':
    main()
