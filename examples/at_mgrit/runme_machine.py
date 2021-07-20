import numpy as np

from pymgrit.core.at_mgrit import AtMgrit
from pymgrit.induction_machine.induction_machine import InductionMachine
from pymgrit.core.mgrit import Mgrit

steps_per_solve = 3
stop_coarse = 1e-06

path_getdp = ''  # path to getdp
path_im3kw = ''  # path to im_3kW


class AtMgritMachine(AtMgrit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_it = []
        self.convergence_criterion(0)

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

    def nested_iteration(self) -> None:
        """
        Generates an initial approximation on the finest grid
        by solving the problem on the coarsest grid and interpolating
        the approximation to the finest level.
        """
        if self.comm_time_rank == 0:
            self.u_coarsest = [self.problem[-1].vector_t_start.clone()]
            for i in range(1, len(self.global_t[-1])):
                self.u_coarsest.append(self.step[-1](u_start=self.u_coarsest[i - 1],
                                                     t_start=self.global_t[-1][i - 1],
                                                     t_stop=self.global_t[-1][i]))
                if self.comm_coarsest_level[i] != self.comm_time_rank:
                    req_s = self.comm_time.send([self.global_t[-1][i], self.u_coarsest[i].pack()],
                                                dest=self.comm_coarsest_level[i],
                                                tag=self.comm_coarsest_level[i])
                else:
                    idx = np.where(self.global_t[-1][i] == self.t[-1])[0][0]
                    self.u[-1][idx] = self.u_coarsest[i]

        else:
            for i in range(self.cpts[-1].size):
                recv = self.comm_time.recv(source=0, tag=self.comm_time_rank)
                idx = np.where(recv[0] == self.t[-1])[0][0]
                self.u[-1][idx].unpack(recv[1])

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl=lvl, cycle_type='V', iteration=0, first_f=True)


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
        self.convergence_criterion(iteration=0)

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


def run_at_mgrit_two_level(k):
    problem = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_start=0, t_stop=0.2, nt=2 ** 14 + 1,
                               path_getdp=path_getdp,
                               path_im3kw=path_im3kw,
                               imposed_speed=2,
                               stop_criterion=1e-06)

    problem1 = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_interval=problem.t[::256],
                                path_getdp=path_getdp,
                                path_im3kw=path_im3kw,
                                imposed_speed=2,
                                steps_per_solve=3,
                                stop_criterion=1e-06)

    solver = AtMgritMachine(problem=[problem, problem1], tol=1, logging_lvl=20, nested_iteration=True, cf_iter=0, k=k)
    solver.solve()


def run_parareal():
    problem = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_start=0, t_stop=0.2, nt=2 ** 14 + 1,
                               path_getdp=path_getdp,
                               path_im3kw=path_im3kw,
                               imposed_speed=2,
                               stop_criterion=1e-06)

    problem1 = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_interval=problem.t[::256],
                                path_getdp=path_getdp,
                                path_im3kw=path_im3kw,
                                imposed_speed=2,
                                steps_per_solve=3,
                                stop_criterion=1e-06)

    solver = MgritMachineConvJl(problem=[problem, problem1], tol=1, logging_lvl=20, nested_iteration=True, cf_iter=0)
    solver.solve()


if __name__ == '__main__':
    run_at_mgrit_two_level(k=16)
    run_at_mgrit_two_level(k=18)
    run_at_mgrit_two_level(k=20)
    run_at_mgrit_two_level(k=22)
    run_at_mgrit_two_level(k=24)

    run_parareal()
