import copy
import numpy as np

from mpi4py import MPI

from pymgrit.core.split import split_communicator
from pymgrit.core.mgrit import Mgrit
from pymgrit.core.vector import Vector as PymgritVector
from pymgrit.core.application import Application

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


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

    def __add__(self, other: 'VectorPetsc') -> 'VectorPetsc':
        """
        Addition of two vector objects (self and other)

        :param other: vector object to be added to self
        :return: sum of vector object self and input object other
        """
        return VectorPetsc(self.get_values() + other.get_values())

    def __sub__(self, other: 'VectorPetsc') -> 'VectorPetsc':
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

    def clone(self) -> 'VectorPetsc':
        """
        Initialize vector object with copied values

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values())

    def clone_zero(self) -> 'VectorPetsc':
        """
        Initialize vector object with zeros

        :rtype: vector object with zero values
        """

        return VectorPetsc(self.get_values() * 0)

    def clone_rand(self) -> 'VectorPetsc':
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
        super(HeatPetsc, self).__init__(*args, **kwargs)
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

        self.dt = self.t[1] - self.t[0]
        self.ksp.setOperators(self.dt * self.space_disc + self.id)

        self.steps = 0
        self.create_time = 0
        self.setOperators_time = 0
        self.solve_time = 0

    def step(self, u_start: VectorPetsc, t_start: float, t_stop: float) -> VectorPetsc:
        """
        Time integration routine for 2D heat equation example problem

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        import time
        start = time.time()
        result = self.dmda.createGlobalVec()
        self.create_time += time.time() - start
        start = time.time()
        if not np.isclose(self.dt, (t_stop - t_start)):
            self.ksp.setOperators((t_stop - t_start) * self.space_disc + self.id)
            self.dt = (t_stop - t_start)
        self.setOperators_time += time.time() - start
        start = time.time()
        self.ksp.solve(self.compute_rhs(u_start=u_start, t_start=t_start, t_stop=t_stop), result)
        self.solve_time += time.time() - start

        self.steps += 1
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


def run_timestepping(nt, space_procs, freq=1, a=1.0, t_start=0, t_stop=1):
    comm_x, comm_t = split_communicator(MPI.COMM_WORLD, space_procs)
    nx = ny = 65
    dmda_coarse = PETSc.DMDA().create([nx, ny], stencil_width=1, comm=comm_x)
    dmda_fine = dmda_coarse.refine()

    t_interval = np.linspace(t_start, t_stop, nt)

    heat_petsc_0 = HeatPetsc(dmda=dmda_fine, comm_x=comm_x, freq=freq, a=a, t_interval=t_interval)
    import time
    start = time.time()
    sol = heat_petsc_0.vector_t_start
    for i in range(1, len(heat_petsc_0.t)):
        sol = heat_petsc_0.step(u_start=sol, t_start=heat_petsc_0.t[i - 1], t_stop=heat_petsc_0.t[i])
    info = {'time_setup': 0, 'time_solve': time.time() - start}

    return info['time_setup'], info['time_solve'], info['time_setup'] + info['time_solve']


def run_mgrit(nt, space_procs, coarsening, freq=1, a=1.0, t_start=0, t_stop=1, cycle='V'):
    size = MPI.COMM_WORLD.Get_size()

    comm_x, comm_t = split_communicator(MPI.COMM_WORLD, space_procs)
    nx = ny = 65
    dmda_coarse = PETSc.DMDA().create([nx, ny], stencil_width=1, comm=comm_x)
    dmda_fine = dmda_coarse.refine()

    t_interval = np.linspace(t_start, t_stop, nt)
    time_procs = int(size / space_procs)

    problem = [HeatPetsc(dmda=dmda_fine, comm_x=comm_x, freq=freq, a=a, t_interval=t_interval)]
    for i in range(0, len(coarsening)):
        problem.append(HeatPetsc(dmda=dmda_fine, comm_x=comm_x, freq=freq, a=a,
                                 t_interval=t_interval[::np.prod(coarsening[:i + 1], dtype=int)]))

    nested_iteration = True if len(coarsening) > 0 else False

    if cycle == 'V':
        mgrit = Mgrit(problem=problem, comm_time=comm_t, comm_space=comm_x, nested_iteration=nested_iteration)
    else:
        mgrit = Mgrit(problem=problem, comm_time=comm_t, comm_space=comm_x, nested_iteration=nested_iteration,
                      cycle_type='F', cf_iter=0)
    info = mgrit.solve()

    return info['time_setup'], info['time_solve'], info['time_setup'] + info['time_solve']


if __name__ == '__main__':
    #setup, solve, setup_and_solve = run_timestepping(nt=2 ** 14 + 1, space_procs=1)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=2)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=4)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=8)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=16)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=32)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 14 + 1, space_procs=64)
    # setup, solve, setup_and_solve =  run_timestepping(nt=2 ** 1, space_procs=space_procs[i], coarsening=item, spatial_coarsening=False,
    #                 cycle='F')

    setup, solve, setup_and_solve = run_mgrit(nt=2 ** 14 + 1, space_procs=4, coarsening=[32, 16, 4, 4], cycle='V')
    # setup, solve, setup_and_solve =  run_mgrit(nt=2 ** 14 + 1, space_procs=4, coarsening=[32, 16, 4, 4], cycle='F')
