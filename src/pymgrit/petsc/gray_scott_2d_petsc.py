"""
Application class for the 2D Gray-Scott equation

Implementation follows:
https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/problem_classes/GrayScott_2D_PETSc_periodic.py
"""

import numpy as np

from mpi4py import MPI

from pymgrit.petsc.vector_petsc import VectorPetsc
from pymgrit.core.application import Application

try:
    from petsc4py import PETSc
except ImportError as e:
    import sys

    sys.exit("This example requires petsc4py.")


class GrayScott(Application):
    """
    Application class implementing the 2D Gray-Scott reaction-diffusion equation with periodic BC
    """

    def __init__(self, dmda: PETSc.DMDA, comm_x: MPI.Comm, du=1.0, dv=0.01, a=0.09, b=0.086, lsol_tol: float = 1e-10,
                 nlsol_tol: float = 1e-10, lsol_maxiter: int = 100, nlsol_maxiter: int = 100,
                 method='IMPL', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dmda = dmda
        self.mx, self.my = self.dmda.sizes
        self.dx = 100.0 / self.mx
        self.dy = 100.0 / self.my
        (self.xs, self.xe), (self.ys, self.ye) = self.dmda.getRanges()
        self.du = du
        self.dv = dv
        self.a = a
        self.b = b
        self.method = method
        self.local_x = dmda.createLocalVec()

        if self.method == 'IMPL' or self.method == 'IMEX':
            pass
        else:
            raise Exception("Unknown method. Choose IMPL (implicit) or IMEX (implicit-explicit)")

        self.space_disc = self.compute_matrix()
        self.id = self.get_id()

        self.vector_template = VectorPetsc(self.dmda.createGlobalVec())
        self.vector_t_start = self.initial_guess()

        if self.method == 'IMEX':
            self.ksp = PETSc.KSP()
            self.ksp.create(comm=comm_x)
            self.ksp.setType('cg')
            pc = self.ksp.getPC()
            pc.setType('none')
            self.ksp.setInitialGuessNonzero(True)
            self.ksp.setFromOptions()
            self.ksp.setTolerances(rtol=lsol_tol, atol=lsol_tol, max_it=lsol_maxiter)
            self.dt = self.t[1] - self.t[0]
            self.ksp.setOperators(self.id - self.dt * self.space_disc)
        else:
            self.snes = PETSc.SNES()
            self.snes.create(comm=comm_x)
            self.snes.setFromOptions()
            self.snes.setTolerances(rtol=nlsol_tol, atol=nlsol_tol, stol=nlsol_tol,
                                    max_it=nlsol_maxiter)

    def step(self, u_start: VectorPetsc, t_start: float, t_stop: float) -> VectorPetsc:
        """
        Time integration routine for 2D Gray Scott example problem

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        result = self.dmda.createGlobalVec()
        if self.method == 'IMEX':
            if not np.isclose(self.dt, (t_stop - t_start)):
                self.ksp.setOperators(self.id - (t_stop - t_start) * self.space_disc)
                self.dt = (t_stop - t_start)
            self.ksp.solve(self.compute_rhs_imex(u_start=u_start, t_start=t_start, t_stop=t_stop), result)
        else:
            self.local_x = self.dmda.createLocalVec()
            self.dt = (t_stop - t_start)
            F = self.dmda.createGlobalVec()
            self.snes.setFunction(self.formFunction, F)
            J = self.dmda.createMatrix()
            self.snes.setJacobian(self.formJacobian, J)
            self.snes.solve(u_start.get_values(), result)
        return VectorPetsc(result)

    def compute_matrix(self) -> PETSc.Mat:
        """
        Helper function to assemble PETSc matrix A

        Returns:
            PETSc matrix object
        """
        space_disc = self.dmda.createMatrix()
        space_disc.setType('aij')
        space_disc.setFromOptions()
        space_disc.setPreallocationNNZ((5, 5))
        space_disc.setUp()

        space_disc.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        for j in range(self.ys, self.ye):
            for i in range(self.xs, self.xe):
                row.index = (i, j)
                row.field = 0
                space_disc.setValueStencil(row, row, self.du * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2))
                row.field = 1
                space_disc.setValueStencil(row, row, self.dv * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2))
                # if j > 0:
                col.index = (i, j - 1)
                col.field = 0
                row.field = 0
                space_disc.setValueStencil(row, col, self.du / self.dy ** 2)
                col.field = 1
                row.field = 1
                space_disc.setValueStencil(row, col, self.dv / self.dy ** 2)
                # if j < my - 1:
                col.index = (i, j + 1)
                col.field = 0
                row.field = 0
                space_disc.setValueStencil(row, col, self.du / self.dy ** 2)
                col.field = 1
                row.field = 1
                space_disc.setValueStencil(row, col, self.dv / self.dy ** 2)
                # if i > 0:
                col.index = (i - 1, j)
                col.field = 0
                row.field = 0
                space_disc.setValueStencil(row, col, self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                space_disc.setValueStencil(row, col, self.dv / self.dx ** 2)
                # if i < mx - 1:
                col.index = (i + 1, j)
                col.field = 0
                row.field = 0
                space_disc.setValueStencil(row, col, self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                space_disc.setValueStencil(row, col, self.dv / self.dx ** 2)
        space_disc.assemble()

        return space_disc

    def get_id(self) -> PETSc.Mat:
        """
        Define identity matrix
        :return: identity matrix
        """
        id = self.dmda.createMatrix()
        id.setType('aij')
        id.setFromOptions()
        id.setPreallocationNNZ((1, 1))
        id.setUp()

        id.zeroEntries()
        row = PETSc.Mat.Stencil()
        for j in range(self.ys, self.ye):
            for i in range(self.xs, self.xe):
                for indx in [0, 1]:
                    row.index = (i, j)
                    row.field = indx
                    id.setValueStencil(row, row, 1.0)

        id.assemble()
        return id

    def compute_rhs_imex(self, u_start: VectorPetsc, t_start: float, t_stop: float) -> PETSc.Vec:
        """
        Right-hand side of spatial system

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: right-hand side of spatial system at each time step in case of implicit time integration
        """
        f = self.dmda.createGlobalVec()
        u = u_start.get_values()

        fa = self.dmda.getVecArray(f)
        xa = self.dmda.getVecArray(u)

        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                fa[i, j, 0] += xa[i, j, 0] - (t_stop - t_start) * xa[i, j, 0] * xa[i, j, 1] ** 2 + (
                        t_stop - t_start) * self.a * (1 - xa[i, j, 0])
                fa[i, j, 1] += xa[i, j, 1] + (t_stop - t_start) * xa[i, j, 0] * xa[i, j, 1] ** 2 - (
                        t_stop - t_start) * self.b * xa[i, j, 1]

        return f

    def formFunction(self, snes, X, F):
        """
        Function to evaluate the residual for the Newton solver
        This function should be equal to the RHS in the solution
        """
        self.dmda.globalToLocal(X, self.local_x)
        x = self.dmda.getVecArray(self.local_x)
        f = self.dmda.getVecArray(F)
        for j in range(self.ys, self.ye):
            for i in range(self.xs, self.xe):
                u = x[i, j]  # center
                u_e = x[i + 1, j]  # east
                u_w = x[i - 1, j]  # west
                u_s = x[i, j + 1]  # south
                u_n = x[i, j - 1]  # north
                u_xx = (u_e - 2 * u + u_w)
                u_yy = (u_n - 2 * u + u_s)
                f[i, j, 0] = x[i, j, 0] - \
                             (self.dt * (self.du * (u_xx[0] / self.dx ** 2 + u_yy[0] / self.dy ** 2) -
                                         x[i, j, 0] * x[i, j, 1] ** 2 + self.a * (1 - x[i, j, 0])))
                f[i, j, 1] = x[i, j, 1] - \
                             (self.dt * (self.dv * (u_xx[1] / self.dx ** 2 + u_yy[1] / self.dy ** 2) +
                                         x[i, j, 0] * x[i, j, 1] ** 2 - self.b * x[i, j, 1]))

    def formJacobian(self, snes, X, J, P):
        """
        Function to return the Jacobian matrix
        """
        self.dmda.globalToLocal(X, self.local_x)
        x = self.dmda.getVecArray(self.local_x)
        P.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        for j in range(self.ys, self.ye):
            for i in range(self.xs, self.xe):
                # diagnoal 2-by-2 block (for u and v)
                row.index = (i, j)
                col.index = (i, j)
                row.field = 0
                col.field = 0
                val = 1.0 - self.dt * (self.du * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2) -
                                       x[i, j, 1] ** 2 - self.a)
                P.setValueStencil(row, col, val)
                row.field = 0
                col.field = 1
                val = self.dt * 2.0 * x[i, j, 0] * x[i, j, 1]
                P.setValueStencil(row, col, val)
                row.field = 1
                col.field = 1
                val = 1.0 - self.dt * (self.dv * (-2.0 / self.dx ** 2 - 2.0 / self.dy ** 2) +
                                       2.0 * x[i, j, 0] * x[i, j, 1] - self.b)
                P.setValueStencil(row, col, val)
                row.field = 1
                col.field = 0
                val = -self.dt * x[i, j, 1] ** 2
                P.setValueStencil(row, col, val)

                # coupling through finite difference part
                col.index = (i, j - 1)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.dt * self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.dt * self.dv / self.dy ** 2)
                col.index = (i, j + 1)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.dt * self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.dt * self.dv / self.dy ** 2)
                col.index = (i - 1, j)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.dt * self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.dt * self.dv / self.dy ** 2)
                col.index = (i + 1, j)
                col.field = 0
                row.field = 0
                P.setValueStencil(row, col, -self.dt * self.du / self.dx ** 2)
                col.field = 1
                row.field = 1
                P.setValueStencil(row, col, -self.dt * self.dv / self.dy ** 2)

        P.assemble()
        if J != P:
            J.assemble()  # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def initial_guess(self) -> VectorPetsc:
        """
        Initial guess
        """
        u = self.dmda.createGlobalVec()
        xa = self.dmda.getVecArray(u)
        for i in range(self.xs, self.xe):
            for j in range(self.ys, self.ye):
                xa[i, j, 0] = 1.0 - 0.5 * np.power(np.sin(np.pi * i * self.dx / 100) *
                                                   np.sin(np.pi * j * self.dy / 100), 100)
                xa[i, j, 1] = 0.25 * np.power(np.sin(np.pi * i * self.dx / 100) *
                                              np.sin(np.pi * j * self.dy / 100), 100)
        return VectorPetsc(u)
