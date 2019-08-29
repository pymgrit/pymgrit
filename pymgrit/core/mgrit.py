from mpi4py import MPI
import time
from operator import itemgetter
from itertools import *
import numpy as np
import logging
import copy
import sys
from typing import Tuple, List
from pymgrit.core import application
from pymgrit.core import grid_transfer


class Mgrit:
    """
    Set up an MGRIT solver, given an array of problems and an array of grid transfers

    The problems are assumed to have a  of constant dimension and the spacetime
    matrix stencil solved is [-Phi  I].
    """

    def __init__(self, problem: List[application.Application], transfer: List[grid_transfer.GridTransfer],
                 it: int = 100, tol: float = 1e-7, nested_iteration: bool = True, cf_iter: int = 1,
                 cycle_type: str = 'V', comm_time: MPI.Comm = MPI.COMM_WORLD, comm_space: MPI.Comm = MPI.COMM_NULL,
                 logging_lvl: int = logging.INFO, output_fcn=None, output_lvl=1) -> None:
        """
        Initialize space-time matrix.
        Phi_args is for any random parameters you may think of later
        :param output_fcn: Function to save results to file.
        :param output_lvl: Output level, possible values 0, 1, 2:
               0 -> output_fcn is never called
               1 -> output_fcn is called when solve stops
               0 -> output_fcn is called after each MGRIT iteration
        :param problem: List of problems per MGRIT level
        :param transfer: List of transfer operators per MGRIT level
        :param it: Max number of iteration
        :param tol: stopping criteria
        :param nested_iteration: With or without nested iteration
        :param cf_iter: Number of CF iteration in MGRIT iteration
        :param cycle_type: 'F' or 'V' cycle
        :param comm_time: Time communicator
        :param comm_space: Space communicator
        :param logging_lvl: Logging level:
               Value <= 10: Debug logging level -> Runtime of all components
               10 > Value <= 20: Info logging level -> Information per MGRIT iteration, summary at the end
        """
        logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S',
                            level=logging_lvl, stream=sys.stdout)

        if len(problem) != (len(transfer) + 1):
            raise Exception('There should be exactly one transfer operator for each level except the coarsest grid')

        for i in range(len(problem) - 1):
            if len(problem[i].t) < len(problem[i + 1].t):
                raise Exception(
                    'The time grid on level ' + str(i + 1) + ' contains more time points than level ' + str(i))

        if cycle_type != 'V' and cycle_type != 'F':
            raise Exception("Cycle-type " + str(cycle_type) + " is not implemented. Choose 'V' or 'F'")

        if output_lvl not in [0, 1, 2]:
            raise Exception("Unknown output level. Choose 0, 1 or 2.")

        for lvl in range(1, len(problem)):
            # if len(problem[lvl - 1].t) / int((len(problem[lvl - 1].t) - 1) / (len(problem[lvl].t) - 1)) > len(
            #         problem[lvl].t):
            #     raise Exception('More C points on level ' + str(lvl - 1) + ' than points on level' + str(lvl))
            for time_point in problem[lvl].t:
                if np.count_nonzero(problem[lvl - 1].t == time_point) != 1:
                    raise Exception(
                        'Point ' + str(time_point) + ' from level ' + str(lvl - 1) + ' is not a point of level ' + str(
                            lvl))

        runtime_setup_start = time.time()
        self.comm_time = comm_time
        self.comm_space = comm_space
        self.comm_time_rank = self.comm_time.Get_rank()
        self.comm_time_size = self.comm_time.Get_size()

        if self.comm_time_size > len(problem[0].t):
            raise Exception('More processors than time points. Not useful and not implemented yet')

        if self.comm_space != MPI.COMM_NULL:
            self.spatial_parallel = True
            self.comm_space_rank = self.comm_space.Get_rank()
            self.comm_space_size = self.comm_space.Get_size()
        else:
            self.spatial_parallel = False
            self.comm_space_rank = -99
            self.comm_space_size = -99

        self.comm_time.barrier()
        self.log_info(f"Start setup")

        self.problem = problem  # List of problems per MGRIT level
        self.lvl_max = len(problem)  # Max level for MGRIT
        self.step = []  # List of time steppers per MGRIT level
        self.u = []  # List of solutions per MGRIT level
        self.v = []  # List of approximate solutions per MGRIT level
        self.g = []  # List of fas right-hand-sides
        self.t = []  # List of time intervals per process per MGRIT level
        self.m = []  # List of coarsening factors
        self.it = it  # Maximal number of iterations
        self.tol = tol  # Convergence tolerance
        self.conv = np.zeros(it + 1)  # Convergence information after each iteration
        self.runtime_solve = 0  # Solve runtime
        self.runtime_setup = 0  # Setup runtime
        self.cf_iter = cf_iter  # Count of C-, F- relaxations
        self.cycle_type = cycle_type  # Cycle type, F or V
        self.restriction = []  # List of restrictions per MGRIT level
        self.interpolation = []  # List of interpolations per MGRIT level
        self.int_start = 0  # First time points of process interval
        self.int_stop = 0  # Last time points of process interval
        self.g_coarsest = []  # Fas residual for the time stepping on coarsest grid
        self.u_coarsest = []  # Solution for the time stepping on coarsest grid
        self.cpts = []  # C-points per process and level corresponding to complete time interval
        self.comm_front = []  # Communication inside F-relax per MGRIT level
        self.comm_back = []  # Communication inside F-relax per MGRIT level
        self.block_size_this_lvl = []  # Block size per process and level with ghost point
        self.index_local_c = []  # Local indices of C-Points
        self.index_local_f = []  # Local indices of F-Points
        self.index_local = []  # Local indices of all points
        self.first_is_f_point = []  # Communication after C-relax
        self.first_is_c_point = []  # Communication after F-relax
        self.last_is_f_point = []  # Communication after F-relax
        self.last_is_c_point = []  # Communication after C-relax
        self.send_to = []  # Which process contains next time point
        self.get_from = []  # Which process contains previous time point
        self.nes_it = nested_iteration  # Local nested iteration value
        self.solve_iter = 0  # The actual MGRIT iteration, for output
        self.output_lvl = output_lvl  # Output level, only 0,1,2

        if output_fcn is not None and callable(output_fcn):
            self.output_fcn = output_fcn
        else:
            self.output_fcn = None

        for lvl in range(self.lvl_max):
            self.t.append(np.copy(problem[lvl].t))
            if lvl != self.lvl_max - 1:
                self.restriction.append(transfer[lvl].restriction)
                self.interpolation.append(transfer[lvl].interpolation)
            if lvl < self.lvl_max - 1:
                self.m.append(int((len(self.problem[lvl].t) - 1) / (len(self.problem[lvl + 1].t) - 1)))
            else:
                self.m.append(1)
            self.setup_points(lvl=lvl)
            self.step.append(problem[lvl].step)
            self.create_u(lvl=lvl)
            if lvl == 0:
                self.v.append(None)
                self.g.append(None)
            else:
                self.v.append([])
                self.g.append([])
                self.v[-1] = [item.init_zero() for item in self.u[lvl]]
                self.g[-1] = [item.init_zero() for item in self.u[lvl]]
            if lvl == self.lvl_max - 1:
                for i in range(len(self.problem[lvl].t)):
                    if i == 0:
                        self.u_coarsest.append(copy.deepcopy(self.problem[lvl].u))
                    else:
                        self.u_coarsest.append(self.problem[lvl].u.init_zero())
                    self.g_coarsest.append(self.problem[lvl].u.init_zero())

        self.setup_comm_info()

        if nested_iteration:
            self.nested_iteration()

        if self.output_fcn is not None and self.output_lvl == 2:
            self.output_fcn(self)

        self.comm_time.barrier()
        self.runtime_setup = time.time() - runtime_setup_start
        self.log_info(f"Setup took {self.runtime_setup} s")

    def log_info(self, message):
        if self.comm_time_rank == 0:
            if self.spatial_parallel:
                if self.comm_space_rank == 0:
                    logging.info(message)
            else:
                logging.info(message)

    def check_input(self):
        pass

    def create_u(self, lvl: int) -> None:
        """

        :param lvl:
        """
        self.u.append([object] * self.block_size_this_lvl[lvl])
        for i in range(len(self.u[lvl])):
            if lvl == 0:
                self.u[lvl][i] = self.problem[lvl].u.init_rand()
            else:
                self.u[lvl][i] = self.problem[lvl].u.init_zero()
        if self.comm_time_rank == 0:
            self.u[lvl][0] = copy.deepcopy(self.problem[lvl].u)

    def iteration(self, lvl: int, cycle_type: str, iteration: int, first_f: bool) -> None:
        """
        :return:
        :param lvl: the corresponding MGRIT level
        :param cycle_type:
        :param iteration:
        :param first_f:
        :return:
        """
        if lvl == self.lvl_max - 1:
            self.forward_solve(lvl=lvl)
            return

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        for i in range(self.cf_iter):
            self.c_relax(lvl=lvl)
            self.c_exchange(lvl=lvl)
            self.f_relax(lvl=lvl)
            self.f_exchange(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        if lvl != 0 and 'F' == cycle_type:
            self.f_exchange(lvl=lvl)
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def f_relax(self, lvl: int) -> None:
        """
        :param lvl: the corresponding MGRIT level
        """
        runtime_f = time.time()
        tmp_send = False
        req_s = None
        rank = self.comm_time_rank
        run = 0
        if len(self.index_local_f[lvl]) > 0:
            for i in np.nditer(self.index_local_f[lvl]):
                if self.comm_front[lvl] and i == np.min(self.index_local_f[lvl]):
                    self.u[lvl][0] = self.comm_time.recv(source=self.get_from[lvl], tag=rank)
                time_loop = time.time()
                if lvl == 0:
                    self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                    t_start=self.t[lvl][i - 1],
                                                    t_stop=self.t[lvl][i])
                else:
                    self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                     t_start=self.t[lvl][i - 1],
                                                                     t_stop=self.t[lvl][i])
                run += (time.time() - time_loop)
                if self.comm_back[lvl] and i == np.max(self.index_local_f[lvl]):
                    tmp_send = True
                    req_s = self.comm_time.isend(self.u[lvl][-1], dest=self.send_to[lvl], tag=self.send_to[lvl])
        if tmp_send:
            req_s.wait()

        logging.debug(
            f"Time per solve {rank} took {run / len(self.index_local_f[lvl]) if len(self.index_local_f[lvl]) > 0 else 0} s")
        # logging.debug(f"F-relax on {rank} took {time.time() - runtime_f} s")

    def c_relax(self, lvl: int) -> None:
        """
        :param lvl: the corresponding MGRIT level
        """
        runtime_c = time.time()
        if len(self.index_local_c[lvl]) > 0:
            for i in np.nditer(self.index_local_c[lvl]):
                if i != 0 or self.comm_time_rank != 0:
                    if lvl == 0:
                        self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                        t_start=self.t[lvl][i - 1],
                                                        t_stop=self.t[lvl][i])
                    else:
                        self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                         t_start=self.t[lvl][i - 1],
                                                                         t_stop=self.t[lvl][i])
        # logging.debug(f"C-relax on {self.comm_time_rank} took {time.time() - runtime_c} s")

    def convergence_criteria(self, it: int) -> None:
        """
        compute pace-time residual
        solve A(u) = g with
             |   I                |
         A = | -Phi   I           |
             |       ...   ...    |
             |            -Phi  I |
        where Phi propagates u_{i-1} from t = t_{i-1} to t = t_i:
           u_i = Phi(u_{i-1}) (including forcing from RHS of PDE)
        and with
           g = (u_0 0 ... 0)^T
        The residual can be computed by
         r_i = Phi(u_{i-1}) - u_i, i = 1, .... nt,
         r_0 = 0
        """
        runtime_conv = time.time()
        r_norm = []

        self.f_exchange(lvl=0)
        self.c_exchange(lvl=0)

        if len(self.index_local_c[0]) > 0:
            for i in np.nditer(self.index_local_c[0]):
                if self.comm_time_rank != 0 or i != 0:
                    r = self.step[0](u_start=self.u[0][i - 1], t_start=self.t[0][i - 1], t_stop=self.t[0][i]) - \
                        self.u[0][i]
                    r_norm.append(r.norm())

        tmp = self.comm_time.allgather(r_norm)
        tmp = [item for sublist in tmp for item in sublist]

        val = 0
        for item in tmp:
            val += item ** 2

        if self.spatial_parallel:
            val = val ** 0.5
            tmp = self.comm_space.allgather(val)

            val = 0
            for item in tmp:
                val += item ** 2

        val = val ** 0.5
        self.conv[it] = val

        # logging.debug(f"Convergence criteria on {self.comm_time_rank} took {time.time() - runtime_conv} s")

    def forward_solve(self, lvl: int) -> None:
        """
        Solve the problem via time stepping on coarsest grid
            :param lvl: the corresponding MGRIT level
        """

        runtime_fs = time.time()
        if self.comm_time_rank == 0:
            for i in range(1, len(self.problem[lvl].t)):
                self.u_coarsest[i] = self.g_coarsest[i] + self.step[lvl](u_start=self.u_coarsest[i - 1],
                                                                         t_start=self.problem[lvl].t[i - 1],
                                                                         t_stop=self.problem[lvl].t[i])
        self.u_coarsest = self.comm_time.bcast(self.u_coarsest, root=0)
        if len(self.cpts[lvl]) > 0:
            self.u[lvl] = [self.u_coarsest[i] for i in self.cpts[lvl]]
            if self.comm_time_rank != 0:
                self.u[lvl] = [self.u[lvl][0]] + self.u[lvl]

        # logging.debug(f"Forward solve on {self.comm_time_rank} took {time.time() - runtime_fs} s")

    def get_c_point(self, lvl: int) -> application.Application:
        """
        Exchange of the first/last C-point between two processes
        :param lvl: the corresponding MGRIT level
        """
        rank = self.comm_time_rank
        tmp_send = False
        tmp = None
        req_s = None

        if self.send_to[lvl + 1] >= 0:
            req_s = self.comm_time.isend(self.u[lvl][self.index_local_c[lvl][-1]], dest=self.send_to[lvl + 1], tag=rank)
            tmp_send = True

        if self.get_from[lvl + 1] >= 0:
            tmp = self.comm_time.recv(source=self.get_from[lvl + 1], tag=self.get_from[lvl + 1])

        if tmp_send:
            req_s.wait()
        return tmp

    def fas_residual(self, lvl: int) -> None:
        """
        Inject the fine grid approximation and its residual to the coarse grid
        :param lvl: the corresponding MGRIT level
        """
        runtime_fas_res = time.time()
        tmp = self.get_c_point(lvl=lvl)
        rank = self.comm_time_rank

        if self.comm_time_rank != 0 and len(self.v[lvl + 1]) > 0:
            # print('tmp on rank', tmp, rank)
            self.v[lvl + 1][0] = self.restriction[lvl](tmp)

        for i in range(len(self.index_local_c[lvl])):
            self.v[lvl + 1][i if rank == 0 else i + 1] = self.restriction[lvl](
                self.u[lvl][self.index_local_c[lvl][i]])

        self.u[lvl + 1] = copy.deepcopy(self.v[lvl + 1])
        if np.size(self.index_local_c[lvl]) > 0:
            for i in range(len(self.index_local_c[lvl])):
                if i != 0 or self.comm_time_rank != 0:
                    if lvl == 0:
                        self.g[lvl + 1][self.index_local[lvl + 1][i]] = \
                            self.restriction[lvl](self.step[lvl](u_start=self.u[lvl][self.index_local_c[lvl][i] - 1],
                                                                 t_start=self.t[lvl][self.index_local_c[lvl][i] - 1],
                                                                 t_stop=self.t[lvl][self.index_local_c[lvl][i]])
                                                  - self.u[lvl][self.index_local_c[lvl][i]]) \
                            + self.v[lvl + 1][self.index_local[lvl + 1][i]] \
                            - self.step[lvl + 1](u_start=self.v[lvl + 1][self.index_local[lvl + 1][i] - 1],
                                                 t_start=self.t[lvl + 1][self.index_local[lvl + 1][i] - 1],
                                                 t_stop=self.t[lvl + 1][self.index_local[lvl + 1][i]])
                    else:
                        self.g[lvl + 1][self.index_local[lvl + 1][i]] = \
                            self.restriction[lvl](self.g[lvl][self.index_local_c[lvl][i]]
                                                  - self.u[lvl][self.index_local_c[lvl][i]]
                                                  + self.step[lvl](u_start=self.u[lvl][self.index_local_c[lvl][i] - 1],
                                                                   t_start=self.t[lvl][self.index_local_c[lvl][i] - 1],
                                                                   t_stop=self.t[lvl][self.index_local_c[lvl][i]])) \
                            + self.v[lvl + 1][self.index_local[lvl + 1][i]] \
                            - self.step[lvl + 1](u_start=self.v[lvl + 1][self.index_local[lvl + 1][i] - 1],
                                                 t_start=self.t[lvl + 1][self.index_local[lvl + 1][i] - 1],
                                                 t_stop=self.t[lvl + 1][self.index_local[lvl + 1][i]])

        if lvl == self.lvl_max - 2:
            tmp_g = self.comm_time.gather([self.g[lvl + 1][i] for i in self.index_local_c[lvl + 1]], root=0)
            tmp_u = self.comm_time.gather([self.u[lvl + 1][i] for i in self.index_local_c[lvl + 1]], root=0)
            if self.comm_time_rank == 0:
                self.g_coarsest = [item for sublist in tmp_g for item in sublist]
                self.u_coarsest = [item for sublist in tmp_u for item in sublist]

        # logging.debug(f"Fas residual on {self.comm_time_rank} took {time.time() - runtime_fas_res} s")

    def nested_iteration(self) -> None:
        """
        Generate initial approximation by the computation and interpolation of approximations on coarser grids
        """
        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl=lvl, cycle_type='V', iteration=0, first_f=True)

    def ouput_run_informations(self):
        msg = ['Run parameter overview \n',
               '{0: <25}'.format(f'interval') + ' : ' + '[' + str(self.problem[0].t[0]) + ', ' + str(
                   self.problem[0].t[-1]) + ']',
               '{0: <25}'.format(f'number points ') + ' : ' + str(len(self.problem[0].t)) + ' points',
               '{0: <25}'.format(f'max dt ') + ' : ' + str(
                   np.max(self.problem[0].t[1:] - self.problem[0].t[:-1])),
               '{0: <25}'.format(f'level') + ' : ' + str(self.lvl_max),
               '{0: <25}'.format(f'coarsening') + ' : ' + str(self.m),
               '{0: <25}'.format(f'cf_iter') + ' : ' + str(self.cf_iter),
               '{0: <25}'.format(f'nested iteration') + ' : ' + str(self.nes_it),
               '{0: <25}'.format(f'cycle type') + ' : ' + str(self.cycle_type),
               '{0: <25}'.format(f'stopping tolerance') + ' : ' + str(self.tol),
               '{0: <25}'.format(f'communicator size time') + ' : ' + str(self.comm_time_size),
               '{0: <25}'.format(f'communicator size space') + ' : ' + str(self.comm_space_size)]
        self.log_info(message='\n'.join(msg))

    def f_exchange(self, lvl: int) -> None:
        """
        Point exchange if the first point is a C-points. Typically, after an F-point update
        :param lvl: the corresponding MGRIT level
        """
        runtime_ex = time.time()
        rank = self.comm_time_rank
        if self.first_is_c_point[lvl]:
            self.u[lvl][0] = self.comm_time.recv(source=self.get_from[lvl], tag=rank)
        if self.last_is_f_point[lvl]:
            self.comm_time.send(self.u[lvl][-1], dest=self.send_to[lvl], tag=self.send_to[lvl])
        # logging.debug(f"Exchange on {self.comm_time_rank} took {time.time() - runtime_ex} s")

    def c_exchange(self, lvl: int) -> None:
        """
        Point exchange if the first point is a F-points. Typically, after an C-point update
        :param lvl: the corresponding MGRIT level
        """
        runtime_ex = time.time()
        rank = self.comm_time_rank
        if self.first_is_f_point[lvl]:
            self.u[lvl][0] = self.comm_time.recv(source=self.get_from[lvl], tag=rank)
        if self.last_is_c_point[lvl]:
            self.comm_time.send(self.u[lvl][-1], dest=self.send_to[lvl], tag=self.send_to[lvl])
        # logging.debug(f"Exchange on {self.comm_time_rank} took {time.time() - runtime_ex} s")

    def solve(self) -> dict:
        """
            :return:
        """
        self.comm_time.barrier()
        self.log_info("Start solve")

        runtime_solve_start = time.time()
        for iteration in range(self.it):

            self.solve_iter = iteration + 1
            time_it_start = time.time()
            self.iteration(lvl=0, cycle_type=self.cycle_type, iteration=iteration, first_f=True)
            self.comm_time.barrier()
            time_it_stop = time.time()
            self.convergence_criteria(it=iteration + 1)

            if iteration == 0:
                self.log_info('{0: <7}'.format(f"iter {iteration + 1}") +
                              '{0: <30}'.format(f" | con: {self.conv[iteration + 1]}") +
                              '{0: <35}'.format(f" | con-fac: -") +
                              '{0: <35}'.format(f" | runtime: {time_it_stop - time_it_start} s"))
            else:
                self.log_info('{0: <7}'.format(f"iter {iteration + 1}") +
                              '{0: <30}'.format(f" | con: {self.conv[iteration + 1]}") +
                              '{0: <35}'.format(f" | con-fac: {self.conv[iteration + 1] / self.conv[iteration]}") +
                              '{0: <35}'.format(f" | runtime: {time_it_stop - time_it_start} s"))

            if self.output_fcn is not None and self.output_lvl == 2:
                self.output_fcn(self)

            if self.conv[iteration + 1] < self.tol:
                break

        self.comm_time.barrier()
        self.runtime_solve = time.time() - runtime_solve_start
        self.log_info(f"Solve took {self.runtime_solve} s")

        if self.output_fcn is not None and self.output_lvl == 1:
            self.output_fcn(self)

        self.ouput_run_informations()
        return {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
                't': self.problem[0].t, 'time_setup': self.runtime_setup}

    def error_correction(self, lvl: int) -> None:
        """
        Compute the coarse grid error approximation
        :param lvl: the corresponding MGRIT level
        """
        for i in range(len(self.index_local_c[lvl])):
            e = self.interpolation[lvl](self.u[lvl + 1][self.index_local[lvl][i]] - self.v[lvl + 1][
                self.index_local[lvl][i]])
            self.u[lvl][self.index_local_c[lvl][i]] = self.u[lvl][self.index_local_c[lvl][i]] + e
        self.c_exchange(lvl=lvl)

    def split_points(self, length: int, size: int, rank: int) -> Tuple[int, int]:
        """
        Split points evenly in size parts and compute the first point and block size of the process interval
        :param length: Number of points
        :param size: Number of processes
        :param rank: Rank
        :return:
        """
        block_size = self.split_into(n=length, p=size)[rank]

        first_i = 0
        if block_size > 0:
            for i in range(size):
                if i == rank:
                    break
                first_i += self.split_into(length, size)[i]
        return block_size, first_i

    @staticmethod
    def split_into(n: int, p: int) -> np.ndarray:
        """
        Split points
        :param n:
        :param p:
        :return:
        """
        return np.array([int(n / p + 1)] * (n % p) + [int(n / p)] * (p - n % p))

    def setup_points(self, lvl: int) -> None:
        """
        Computes grid information per process
        :param lvl: the corresponding MGRIT level
        :return:
        """
        nt = np.size(self.problem[lvl].t)
        all_pts = np.array(range(0, nt))

        # Compute all pts per process
        # First level by splitting of points
        # other levels by time depending on first grid
        if lvl == 0:
            block_size_this_lvl, first_i_this_lvl = self.split_points(length=np.size(all_pts),
                                                                      size=self.comm_time_size,
                                                                      rank=self.comm_time_rank)
            all_pts = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
            self.int_start = self.problem[lvl].t[all_pts[0]]
            self.int_stop = self.problem[lvl].t[all_pts[-1]]
        else:
            all_pts = np.where((self.problem[lvl].t >= self.int_start) & (self.problem[lvl].t <= self.int_stop))[0]

        # Compute C- and F-points
        all_cpts = np.array(range(0, nt, self.m[lvl]))
        all_fpts = np.array(list(set(np.array(range(0, nt))) - set(all_cpts)))
        cpts = np.sort(np.array(list(set(all_pts) - set(all_fpts)), dtype='int'))
        fpts = np.array(list(set(all_pts) - set(cpts)))
        fpts2 = np.array([item for sublist in np.array([np.array(xi) for xi in np.asarray(
            [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(fpts), lambda x: x[0] - x[1])])])[::-1] for item
                          in
                          sublist])

        # Communication in F-relax
        need_communication_front = False
        need_communication_back = False

        if np.size(fpts) > 0 and fpts[np.argmin(fpts)] - 1 in all_fpts:
            need_communication_front = True
        if np.size(fpts) > 0 and fpts[np.argmax(fpts)] + 1 in all_fpts:
            need_communication_back = True

        # Communication before and after C- and F-relax
        if len(all_pts) > 0 and all_pts[0] in cpts and all_pts[0] != 0:
            first_is_c_point = True
        else:
            first_is_c_point = False
        if len(all_pts) > 0 and all_pts[0] in fpts2 and all_pts[0] - 1 in all_cpts:
            first_is_f_point = True
        else:
            first_is_f_point = False
        if len(all_pts) > 0 and all_pts[-1] in cpts and all_pts[-1] != nt - 1:
            last_is_c_point = True
        else:
            last_is_c_point = False
        if len(all_pts) > 0 and all_pts[-1] in fpts2 and all_pts[-1] != nt - 1 and all_pts[-1] + 1 in all_cpts:
            last_is_f_point = True
        else:
            last_is_f_point = False

        # Add ghost point if needed, set time interval
        if self.comm_time_rank != 0 and len(all_pts) > 0:
            tmp = np.zeros(len(all_pts) + 1, dtype=int)
            tmp[0] = all_pts[0] - 1
            tmp[1:] = all_pts
            all_pts_with_ghost = tmp
        else:
            all_pts_with_ghost = all_pts

        self.t[lvl] = self.problem[lvl].t[all_pts_with_ghost]

        # Setup local indices
        index_local_c = np.zeros_like(cpts)
        index_local_f = np.zeros_like(fpts2)
        index_local = np.zeros_like(all_pts)

        for i in range(len(cpts)):
            index_local_c[i] = np.where(cpts[i] == all_pts_with_ghost)[0]

        for i in range(len(fpts2)):
            index_local_f[i] = np.where(fpts2[i] == all_pts_with_ghost)[0]

        for i in range(len(all_pts)):
            index_local[i] = np.where(all_pts[i] == all_pts_with_ghost)[0]

        self.cpts.append(cpts)
        self.comm_front.append(need_communication_front)
        self.comm_back.append(need_communication_back)
        self.block_size_this_lvl.append(len(all_pts_with_ghost))
        self.index_local.append(index_local)
        self.index_local_c.append(index_local_c)
        self.index_local_f.append(index_local_f)
        self.first_is_c_point.append(first_is_c_point)
        self.first_is_f_point.append(first_is_f_point)
        self.last_is_c_point.append(last_is_c_point)
        self.last_is_f_point.append(last_is_f_point)

    def setup_comm_info(self) -> None:
        """
        Computes which process holds the previous and next point for each lvl and process
        """
        start = np.zeros(self.comm_time_size)
        stop = np.zeros(self.comm_time_size)
        for lvl in range(self.lvl_max):
            nt = np.size(self.problem[lvl].t)
            all_pts = np.array(range(0, nt))
            this_level = np.zeros(0, dtype=int)
            for proc in range(self.comm_time_size):
                if lvl == 0:
                    block_size_this_lvl, first_i_this_lvl = self.split_points(length=np.size(all_pts),
                                                                              size=self.comm_time_size,
                                                                              rank=proc)
                    tmp = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
                    start[proc] = self.problem[lvl].t[tmp[0]]
                    stop[proc] = self.problem[lvl].t[tmp[-1]]
                else:
                    tmp = np.where((self.problem[lvl].t >= start[proc]) & (self.problem[lvl].t <= stop[proc]))[0]
                this_level = np.hstack((this_level, np.ones(len(tmp)) * proc)).astype(int)
            points = np.where(self.comm_time_rank == this_level)[0]
            if len(points) > 0:
                if points[0] == 0:
                    front = -99
                else:
                    front = this_level[points[0] - 1]
                if points[-1] == nt - 1:
                    back = -99
                else:
                    back = this_level[points[-1] + 1]
            else:
                front = -99
                back = -99
            self.send_to.append(back)
            self.get_from.append(front)
