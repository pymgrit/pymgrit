import pathlib
import os
import numpy as np

from pymgrit.core.mgrit import Mgrit
from pymgrit.induction_machine.induction_machine import InductionMachine
from pymgrit.induction_machine.grid_transfer_machine import GridTransferMachine
from pymgrit.core.grid_transfer_copy import GridTransferCopy


class MgritMachineConvJl(Mgrit):
    """
    MGRIT optimized for the getdp induction machine
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        MGRIT optimized for the getdp induction machine

        :param compute_f_after_convergence: computes solution of F-points at the end
        """
        super().__init__(*args, **kwargs)
        self.last_it = []
        self.convergence_criteria(iteration=0)

    def nested_iteration(self) -> None:
        """
        Generates an initial approximation on the finest grid
        by solving the problem on the coarsest grid and interpolating
        the approximation to the finest level.

        Performs nested iterations with a sin rhs
        """
        change = False
        tmp_problem_pwm = np.zeros(len(self.problem))
        if self.problem[0].pwm:
            change = True
            for lvl in range(len(self.problem)):
                tmp_problem_pwm[lvl] = self.problem[lvl].pwm
                self.problem[lvl].fopt[-1] = 0

        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl, 'V', 0, True)

        if change:
            for lvl in range(len(self.problem)):
                self.problem[lvl].fopt[-1] = tmp_problem_pwm[lvl]

    def iteration(self, lvl: int, cycle_type: str, iteration: int, first_f: bool):
        """
        MGRIT iteration on level lvl

        :param lvl: MGRIT level
        :param cycle_type: Cycle type
        :param iteration: Number of current iteration
        :param first_f: F-relaxation at the beginning
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for _ in range(self.cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        if lvl > 0:
            self.f_relax(lvl=lvl)

        if lvl != 0 and cycle_type == 'F':
            self.f_exchange(lvl=lvl)
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def convergence_criterion(self, iteration: int) -> None:
        """
        Maximum norm of all C-points

        :param iteration: iteration number
        """
        if len(self.last_it) != len(self.index_local_c[0]):
            self.last_it = np.zeros(len(self.index_local_c[0]))
        new = np.zeros_like(self.last_it)
        j = 0
        tmp = 0
        if self.index_local_c[0].size > 0:
            for i in np.nditer(self.index_local_c[0]):
                new[j] = self.u[0][i].jl
                j = j + 1
            tmp = 100 * np.max(
                np.abs(np.abs(np.divide((new - self.last_it), new, out=np.zeros_like(self.last_it), where=new != 0))))

        tmp = self.comm_time.allgather(tmp)
        self.conv[iteration] = np.max(np.abs(tmp))
        self.last_it = np.copy(new)


def run_F_FCF_17_4_4_4_4():
    def output_fcn(self):
        now = 'F_FCF_17_4_4_4_4'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)
    machine_2 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 6 + 1)
    machine_3 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 4 + 1)
    machine_4 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4]
    transfer = [
        GridTransferMachine(fine_grid='im_3kW_17k', coarse_grid='im_3kW_4k',
                            path_meshes=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/'),
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn,
                               tol=1, cycle_type='F', cf_iter=1)
    mgrit.solve()


def run_F_FCF_17_17_17_17_17():
    def output_fcn(self):
        now = 'F_FCF_17_17_17_17_17'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)
    machine_2 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 6 + 1)
    machine_3 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 4 + 1)
    machine_4 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4]
    transfer = [
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn, tol=1,
                               cycle_type='F', cf_iter=1)
    mgrit.solve()


def run_V_F_17_17():
    def output_fcn(self):
        now = 'V_F_17_17'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)

    problem = [machine_0, machine_1]
    transfer = [
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn, tol=1,
                               cycle_type='V', cf_iter=0)
    mgrit.solve()


def run_V_FCF_17_4():
    def output_fcn(self):
        now = 'V_FCF_17_4'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)

    problem = [machine_0, machine_1]
    transfer = [
        GridTransferMachine(fine_grid='im_3kW_17k', coarse_grid='im_3kW_4k',
                            path_meshes=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/')
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn, tol=1,
                               cycle_type='V', cf_iter=1)
    mgrit.solve()


def run_V_FCF_17_4_4_4_4():
    def output_fcn(self):
        now = 'V_FCF_17_4_4_4_4'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)
    machine_2 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 6 + 1)
    machine_3 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 4 + 1)
    machine_4 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_4k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4]
    transfer = [
        GridTransferMachine(fine_grid='im_3kW_17k', coarse_grid='im_3kW_4k',
                            path_meshes=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/'),
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn,
                               tol=1, cycle_type='V', cf_iter=1)
    mgrit.solve()


def run_V_FCF_17_17():
    def output_fcn(self):
        now = 'V_FCF_17_17'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)

    problem = [machine_0, machine_1]
    transfer = [
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn, tol=1,
                               cycle_type='V', cf_iter=1)
    mgrit.solve()


def run_V_FCF_17_17_17_17_17():
    def output_fcn(self):
        now = 'V_FCF_17_17_17_17_17'
        pathlib.Path('results/' + now + '/' + str(self.solve_iter)).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        tr = [self.u[0][i].tr for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr,
               'time': self.runtime_solve, 'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.solve_iter) + '/' + str(self.t[0][-1]), sol)

    nonlinear = True
    pwm = True
    t_start = 0
    t_stop = 0.01025390625

    # Complete
    machine_0 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=10753)

    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::42], np.arange(0, len(machine_0.t))[::42][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)

    machine_1 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 8 + 1)
    machine_2 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 6 + 1)
    machine_3 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 4 + 1)
    machine_4 = InductionMachine(nonlinear=nonlinear,
                                 pwm=pwm,
                                 t_start=t_start,
                                 t_stop=t_stop,
                                 grid='im_3kW_17k',
                                 path_im3kw=os.getcwd() + '/../../src/pymgrit/induction_machine/im_3kW/',
                                 path_getdp=os.getcwd() + '/../../src/pymgrit/induction_machine/getdp/getdp',
                                 nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4]
    transfer = [
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy(),
        GridTransferCopy()
    ]

    mgrit = MgritMachineConvJl(problem=problem, transfer=transfer, nested_iteration=True, output_fcn=output_fcn, tol=1,
                               cycle_type='V', cf_iter=1)
    mgrit.solve()


if __name__ == '__main__':
    run_V_F_17_17()
