"""
MGRIT solver in FAS formulation
"""
import time
import copy
import logging
import sys
from operator import itemgetter
from typing import Tuple, List
from itertools import groupby
import numpy as np

from mpi4py import MPI

from pymgrit.core.application import Application
from pymgrit.core.grid_transfer import GridTransfer
from pymgrit.core.grid_transfer_copy import GridTransferCopy


class Mgrit:
    """
    MGRIT solver class

    Implementation of MGRIT FAS algorithm for solving
    time-stepping problems of the form
      u_i = Phi(u_{i-1}),
    where Phi propagates u_{i-1} from t = t_{i-1} to t = t_i.

    It is assumed that the problems have a constant dimension
    and the solved space-time matrix stencil is [-Phi I].
    """

    def __init__(self, problem: List[Application], transfer: List[GridTransfer] = None, weight_c: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-7, nested_iteration: bool = True, cf_iter: int = 1,
                 cycle_type: str = 'V', comm_time: MPI.Comm = MPI.COMM_WORLD, comm_space: MPI.Comm = MPI.COMM_NULL,
                 logging_lvl: int = logging.INFO, output_fcn=None, output_lvl=1, t_norm=2,
                 random_init_guess: bool = False, conv_crit: int = 0) -> None:
        """
        Initialize MGRIT solver.

        :param problem: List of problems (one for each MGRIT level)
        :param weight_c: C-relaxation weight
        :param transfer: List of spatial transfer operators (one for each pair of consecutive MGRIT levels)
        :param max_iter: Maximum number of iterations
        :param tol: stopping tolerance
        :param nested_iteration: With (True) or without (False) nested iterations
        :param cf_iter: Number of CF relaxations in each MGRIT iteration
        :param cycle_type: 'F' or 'V' cycle
        :param comm_time: Time communicator
        :param comm_space: Space communicator
        :param logging_lvl: Logging level:
               Value = 10: Debug logging level -> Runtime of all components
               Value = 20: Info logging level  -> Information per MGRIT iteration + summary at the end
               Value = 30: No logging level    -> No information
        :param output_fcn: Function for saving solution values to file
        :param output_lvl: Output level, possible values 0, 1, 2:
               0 -> output_fcn is never called
               1 -> output_fcn is called at the end of the simulation
               2 -> output_fcn is called after each MGRIT iteration
        :param random_init_guess: Use (True) or do not use (False) random initial guess
        :param t_norm: Norm in time (only for global convergence criterion):
               1 -> One norm
               2 -> Two norm
               3 -> Inf norm
        :param conv_crit: Convergence criterion, possible values 0, 1, 2, 3:
               0 -> global space-time residual
               1 -> global jump
               2 -> local space-time residual
               3 -> local jump
        """
        logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%m-%y %H:%M:%S',
                            level=logging_lvl, stream=sys.stdout)

        # Set standard grid transfer operators if no transfer operators are given
        if transfer is None:
            transfer = [GridTransferCopy() for _ in range(len(problem) - 1)]

        # Check input parameters
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
            if len(set(problem[lvl - 1].t.tolist()).intersection(set(problem[lvl].t.tolist()))) != len(problem[lvl].t):
                raise Exception(
                    'Some points from level ' + str(lvl - 1) + ' are not points of level ' + str(lvl))

        if t_norm not in [1, 2, 3]:
            raise Exception(
                'Unknown norm. Please choose 1 (one norm), 2 (two-norm) or 3 (inf-norm)')

        if conv_crit not in [0, 1, 2, 3]:
            raise Exception(
                'Unknown convergence criterion. Please choose: '
                '0 (global space-time residual), '
                '1 (global jump)'
                '2 (local space-time residual)'
                '3 (local jump)')

        if isinstance(cf_iter, int):
            cf_iter = [cf_iter for _ in range(len(problem))]
        elif isinstance(cf_iter, list):
            if len(cf_iter) < len(problem) - 1:
                raise Exception(
                    'Too few cf_iter. '
                    'Specify a list of values for all but the coarsest level or an integer (used for all levels).')
        else:
            raise Exception(
                'Incorrect datatype cf_iter. '
                'Specify a list of values for all but the coarsest level or an integer ( used for all levels).')

        self.comm_time = comm_time
        self.comm_space = comm_space
        self.comm_time_rank = self.comm_time.Get_rank()
        self.comm_time_size = self.comm_time.Get_size()

        if self.comm_time_size > len(problem[0].t):
            raise Exception('More processors than time points. Not useful and not implemented yet')

        # Check if spatial parallelism is used
        if self.comm_space != MPI.COMM_NULL:
            self.spatial_parallel = True
            self.comm_space_rank = self.comm_space.Get_rank()
            self.comm_space_size = self.comm_space.Get_size()
        else:
            self.spatial_parallel = False
            self.comm_space_rank = -99
            self.comm_space_size = 1

        # Start timer for setup time
        self.comm_time.barrier()
        runtime_setup_start = time.time()
        self.log_info(f"Start setup")

        # Initialize MGRIT parameters
        self.problem = problem  # List of problems (one per MGRIT level)
        self.weight_c = weight_c  # C-relaxation weight
        self.lvl_max = len(problem)  # Max number of MGRIT levels
        self.step = []  # List of time integration routines (one per MGRIT level)
        self.u = []  # List of solutions (one per MGRIT level)
        self.v = []  # List of restricted unknowns (one per MGRIT level)
        self.g = []  # List of FAS right-hand sides (one per MGRIT level)
        self.t = []  # List of local time intervals (one per MGRIT level)
        self.m = []  # List of coarsening factors
        self.restriction = []  # List of restriction operators (one per MGRIT level - except for coarsest)
        self.interpolation = []  # List of interpolation operators (one per MGRIT level - except for coarsest)
        self.tol = tol  # Convergence tolerance
        self.conv = np.zeros(max_iter + 1)  # Convergence information after each iteration
        self.cf_iter = cf_iter  # Number of CF-relaxations
        self.cycle_type = cycle_type  # Cycle type, F or V
        self.random_init_guess = random_init_guess  # Random initial guess
        self.iter_max = max_iter  # Maximum number of iterations
        self.solve_iter = 0  # MGRIT iteration number; for output
        self.nes_it = nested_iteration  # Local nested iteration value
        self.runtime_solve = 0  # Solve runtime
        self.runtime_setup = 0  # Setup runtime
        self.int_start = 0  # Index of first time point of local time interval
        self.int_stop = 0  # Index of last time points of local time interval
        self.cpts = []  # Global index of local C-points
        self.index_local_c = []  # Local indices of C-Points
        self.index_local_f = []  # Local indices of F-Points
        self.index_local = []  # Local indices of all points
        self.comm_front = []  # Communication inside F-relax per MGRIT level
        self.comm_back = []  # Communication inside F-relax per MGRIT level
        self.first_is_f_point = []  # Communication after C-relax
        self.first_is_c_point = []  # Communication after F-relax
        self.last_is_f_point = []  # Communication after F-relax
        self.last_is_c_point = []  # Communication after C-relax
        self.send_to = []  # Which process contains next time point
        self.get_from = []  # Which process contains previous time point
        self.global_t = []  # Global time information
        self.t_norm = 1 if t_norm == 1 else None if t_norm == 2 else np.inf  # Time norm
        self.conv_crit = conv_crit
        self.global_conv_crit = True if (self.conv_crit == 0 or self.conv_crit == 1) else False
        if not self.global_conv_crit:
            self.log_info(
                f"A local criterion is used. The following output describes only the convergence of "
                f"the time points of one process")
        self.save_values_last_iter = None

        self.requests = []
        self.tags_send = [np.array([50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]) for _ in
                          range(self.lvl_max)]
        self.tags_recv = [np.array([50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]) for _ in
                          range(self.lvl_max)]
        self.sender_finished = [[False for _ in range(8)] for _ in range(self.lvl_max)]

        # Set output level and output function
        self.output_lvl = output_lvl  # Output level, only 0,1,2
        if output_fcn is not None and callable(output_fcn):
            self.output_fcn = output_fcn
        else:
            self.output_fcn = None

        # Set local MGRIT parameters
        for lvl in range(self.lvl_max):
            self.t.append(np.copy(problem[lvl].t))
            if lvl != self.lvl_max - 1:
                self.restriction.append(transfer[lvl].restriction)
                self.interpolation.append(transfer[lvl].interpolation)
            if lvl < self.lvl_max - 1:
                tmp_cpts = np.where(np.in1d(self.problem[lvl].t, self.problem[lvl + 1].t))[0]
                tmp_m = np.diff(tmp_cpts)[0]
                self.m.append(int(tmp_m))
                if not np.all(np.isclose(np.diff(tmp_cpts), np.diff(tmp_cpts)[0])) and self.comm_time_rank == 0:
                    logging.warning('Non-uniform coarsening between level ' + str(lvl) + ' and ' + str(
                        lvl + 1) + '. Poorly tested.')
            else:
                self.m.append(1)
            self.setup_points_and_comm_info(lvl=lvl)
            self.step.append(problem[lvl].step)
            self.create_u_v_g(lvl=lvl)

        self.finished = [False, None]
        if self.comm_time_rank != 0:
            self.pre_finished = [False, None]
        else:
            self.pre_finished = [True, 0]

        # Use or do not use nested iteration
        if nested_iteration:
            self.nested_iteration()

        if self.conv_crit == 1 or self.conv_crit == 3:
            self.save_values_last_iter = [item.clone() for item in self.u[0]]

        # Stop timer for setup time
        if self.iter_max == 0:
            self.comm_time.barrier()
        self.runtime_setup = time.time() - runtime_setup_start

        if self.output_fcn is not None and self.output_lvl == 2:
            self.output_fcn(self)

        self.log_info(f"Setup took {self.runtime_setup} s")

    def log_info(self, message: str) -> None:
        """
        Writes a message to the logger.
        Only one process

        :param message: Message
        """
        if self.comm_time_rank == self.comm_time_size - 1:
            if self.spatial_parallel:
                if self.comm_space_rank == 0:
                    logging.info(message)
            else:
                logging.info(message)

    def iteration(self, lvl: int, cycle_type: str, iteration: int, first_f: bool) -> None:
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

        if (lvl > 0 or (iteration == 0 and lvl == 0)) and first_f:
            self.f_relax(lvl=lvl)

        for _ in range(self.cf_iter[lvl]):
            self.c_relax(lvl=lvl)
            self.f_relax(lvl=lvl)

        self.fas_residual(lvl=lvl)

        self.iteration(lvl=lvl + 1, cycle_type=cycle_type, iteration=iteration, first_f=True)

        self.error_correction(lvl=lvl)

        self.f_relax(lvl=lvl)

        if lvl != 0 and cycle_type == 'F':
            self.iteration(lvl=lvl, cycle_type='V', iteration=iteration, first_f=False)

    def f_relax(self, lvl: int) -> None:
        """
        F-relaxation on level lvl.

        F-relaxation updates solution values at F-points by propagating the solution
        from one C-point to all F-points up to the next C-point.

        :param lvl: MGRIT level
        """
        logging.debug(f"Start f_relax on {self.comm_time_rank}")
        runtime_f = time.time()

        # Send data if the last point of the process is a C-point
        if self.last_is_c_point[lvl]:
            self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=0)

        # Receive data if the first point is an F-point and the first point of an F-interval
        if self.first_is_f_point[lvl] and not self.sender_finished[lvl][0]:
            self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=0))

        # Iterate over F-points
        if self.index_local_f[lvl].size > 0:
            for i in np.nditer(self.index_local_f[lvl]):
                # Receive data if the previous F-point is on another process
                if self.comm_front[lvl] and i == np.min(self.index_local_f[lvl]) and not self.sender_finished[lvl][1]:
                    self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=1))

                # Perform time integration, consider FAS rhs on lvl > 0.
                if lvl == 0:
                    self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                    t_start=self.t[lvl][i - 1],
                                                    t_stop=self.t[lvl][i])
                else:
                    self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                     t_start=self.t[lvl][i - 1],
                                                                     t_stop=self.t[lvl][i])

                # Send data if the next F-point is on another process
                if self.comm_back[lvl] and i == np.max(self.index_local_f[lvl]):
                    self.send(data=self.u[lvl][i].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=1)

        logging.debug(f"F-relax on {self.comm_time_rank} took {time.time() - runtime_f} s")

    def c_relax(self, lvl: int) -> None:
        """
        C-relaxation on level lvl.

        C-relaxation updates solution values at C-points by propagating the
        solution from the preceeding F-points.

        :param lvl: MGRIT level
        """
        runtime_c = time.time()

        # Send data if the last point of the process is an F-point and the last point of an F-interval
        if self.last_is_f_point[lvl]:
            self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=2)

        # Receive data if the first point is a C-point
        if self.first_is_c_point[lvl] and not self.sender_finished[lvl][2]:
            self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=2))

        # Iterate over C-points
        if self.index_local_c[lvl].size > 0:
            for i in np.nditer(self.index_local_c[lvl]):
                if i != 0 or self.comm_time_rank != 0:
                    # Perform time integration, consider FAS rhs on lvl > 0
                    if lvl == 0:
                        self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                        t_start=self.t[lvl][i - 1],
                                                        t_stop=self.t[lvl][i]) * self.weight_c + \
                                         self.u[lvl][i] * (1.0 - self.weight_c)
                    else:
                        self.u[lvl][i] = (self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                          t_start=self.t[lvl][i - 1],
                                                                          t_stop=self.t[lvl][i])) * self.weight_c + \
                                         self.u[lvl][i] * (1.0 - self.weight_c)

        logging.debug(f"C-relax on {self.comm_time_rank} took {time.time() - runtime_c} s")

    def compute_jump(self) -> list:
        """

        :return:
        """
        jump_norm = []
        # Iterate over C-points
        if self.index_local_c[0].size > 0:
            for i in np.nditer(self.index_local_c[0]):
                if self.comm_time_rank != 0 or i != 0:
                    jump = self.u[0][i] - self.save_values_last_iter[i]
                    jump_norm.append(jump.norm())
        self.save_values_last_iter = [item.clone() for item in self.u[0]]
        return jump_norm

    def compute_residual(self) -> list:
        """
        Computes the space-time residual
          r_i = Phi(u_{i-1}) - u_i, i = 1, .... nt,
          r_0 = 0

        :return: list of space-time residuals
        """
        r_norm = []

        # Send data if the last point of the process is an F-point and the last point of an F-interval
        if self.last_is_f_point[0]:
            self.send(data=self.u[0][-1].pack(), dest=self.send_to[0], lvl=0, op_id=7)

        # Receive data if the first point is a C-point
        if self.first_is_c_point[0] and not self.sender_finished[0][7]:
            self.u[0][0].unpack(self.receive(source=self.get_from[0], lvl=0, op_id=7))

        # Iterate over C-points
        if self.index_local_c[0].size > 0:
            for i in np.nditer(self.index_local_c[0]):
                if self.comm_time_rank != 0 or i != 0:
                    residual = self.step[0](u_start=self.u[0][i - 1],
                                            t_start=self.t[0][i - 1],
                                            t_stop=self.t[0][i]) - self.u[0][i]
                    r_norm.append(residual.norm())
        return r_norm

    def convergence_criterion(self, iteration: int) -> None:
        """
        Stopping criterion based on global or local criterion

        :param iteration: MGRIT iteration number
        """
        runtime_conv = time.time()

        if self.conv_crit == 0 or self.conv_crit == 1:
            if self.conv_crit == 0:
                val = self.compute_residual()
            else:
                val = self.compute_jump()
            tmp = MPI.COMM_WORLD.gather(val, root=0)
            if self.comm_time_rank == 0:
                self.conv[iteration] = np.linalg.norm(np.array([item for sublist in tmp for item in sublist]),
                                                      ord=self.t_norm)
            self.conv[iteration] = MPI.COMM_WORLD.bcast(self.conv[iteration], root=0)

        elif self.conv_crit == 2 or self.conv_crit == 3:

            if self.conv_crit == 2:
                val = self.compute_residual()
            else:
                val = self.compute_jump()

            # Receive convergence information from last process
            if self.comm_time_rank > 0 and not self.sender_finished[0][6]:
                self.pre_finished[0] = self.receive(source=self.comm_time_rank - 1, lvl=0, op_id=6)
                if self.pre_finished[0]:
                    self.pre_finished[1] = iteration

            self.finished[0] = self.pre_finished[0] and (
                    all(i < self.tol for i in val) or iteration == self.iter_max)
            self.finished[1] = iteration

            # Send convergence information to the next process
            if self.comm_time_rank < self.comm_time_size - 1:
                self.send(data=self.finished[0], dest=self.comm_time_rank + 1, lvl=0, op_id=6)

            self.conv[iteration] = np.linalg.norm(val, ord=self.t_norm)

        logging.debug(f"Convergence criterion on {self.comm_time_rank} took {time.time() - runtime_conv} s")

    def forward_solve(self, lvl: int) -> None:
        """
        Solves the problem directly on level lvl with time stepping.

        :param lvl: MGRIT level
        """
        runtime_fs = time.time()

        # Receive data if another process holds an earlier time point
        if self.get_from[lvl] != -99 and not self.sender_finished[lvl][5]:
            self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=5))

        # Iterate over all points
        for i in range(1, len(self.u[lvl])):
            # Perform time integration, consider FAS rhs on lvl > 0
            if lvl == 0:
                self.u[lvl][i] = self.step[lvl](u_start=self.u[lvl][i - 1],
                                                t_start=self.t[lvl][i - 1],
                                                t_stop=self.t[lvl][i])
            else:
                self.u[lvl][i] = self.g[lvl][i] + self.step[lvl](u_start=self.u[lvl][i - 1],
                                                                 t_start=self.t[lvl][i - 1],
                                                                 t_stop=self.t[lvl][i])
        # Send data when another process holds a later time point
        if self.send_to[lvl] != -99:
            self.send(data=self.u[lvl][self.index_local[lvl][-1]].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=5)

        logging.debug(f"Forward solve on {self.comm_time_rank} took {time.time() - runtime_fs} s")

    def fas_residual(self, lvl: int) -> None:
        """
        Injects the fine-grid approximation and its residual
        from level lvl to the next coarser grid.

        :param lvl: MGRIT level
        """
        runtime_fas_res = time.time()

        # Restrict all C-points
        for i in range(len(self.index_local_c[lvl])):
            self.u[lvl + 1][i if self.comm_time_rank == 0 else i + 1] = self.restriction[lvl](
                self.u[lvl][self.index_local_c[lvl][i]])

        # Send data if the last point of the process is an F-point and the last point of an F-interval
        if self.last_is_f_point[lvl]:
            self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=3)

        # Receive data if the first point is a C-point
        if self.first_is_c_point[lvl] and not self.sender_finished[lvl][3]:
            self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=3))

        # Send data if not the last process that has a time point on lvl+1
        if self.send_to[lvl + 1] >= 0:
            self.send(data=self.u[lvl + 1][self.index_local[lvl + 1][-1]].pack(), dest=self.send_to[lvl + 1], lvl=lvl,
                      op_id=4)

        # Receive data if not the first process that has a time point on lvl+1
        if self.get_from[lvl + 1] >= 0 and not self.sender_finished[lvl][4]:
            self.u[lvl + 1][0].unpack(self.receive(source=self.get_from[lvl + 1], lvl=lvl, op_id=4))

        # Clone all C-points
        self.v[lvl + 1] = [item.clone() for item in self.u[lvl + 1]]

        # Iterate over C-points
        if np.size(self.index_local_c[lvl]) > 0:
            for i in range(len(self.index_local_c[lvl])):
                if i != 0 or self.comm_time_rank != 0:
                    # Compute fas rhs
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

        logging.debug(f"Fas residual on {self.comm_time_rank} took {time.time() - runtime_fas_res} s")

    def nested_iteration(self) -> None:
        """
        Generates an initial approximation on the finest grid
        by solving the problem on the coarsest grid and interpolating
        the approximation to the finest level.
        """
        self.forward_solve(self.lvl_max - 1)

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                if i != 0 or self.comm_time_rank != 0:
                    self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                        u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            if lvl > 0:
                self.iteration(lvl=lvl, cycle_type='V', iteration=0, first_f=True)

    def ouput_run_information(self) -> None:
        """
        Outputs information of pyMGRIT run.
        """
        msg = ['Run parameter overview',
               '  ' + '{0: <25}'.format(f'time interval') + ' : ' + '[' + str(self.problem[0].t[0]) + ', ' + str(
                   self.problem[0].t[-1]) + ']',
               '  ' + '{0: <25}'.format(f'number of time points ') + ' : ' + str(len(self.problem[0].t)),
               '  ' + '{0: <25}'.format(f'max dt ') + ' : ' + str(
                   np.max(self.problem[0].t[1:] - self.problem[0].t[:-1])),
               '  ' + '{0: <25}'.format(f'number of levels') + ' : ' + str(self.lvl_max),
               '  ' + '{0: <25}'.format(f'coarsening factors') + ' : ' + str(self.m[:-1]),
               '  ' + '{0: <25}'.format(f'relaxation weight') + ' : ' + str(self.weight_c),
               '  ' + '{0: <25}'.format(f'cf_iter') + ' : ' + str(self.cf_iter[:self.lvl_max - 1]),
               '  ' + '{0: <25}'.format(f'nested iteration') + ' : ' + str(self.nes_it),
               '  ' + '{0: <25}'.format(f'cycle type') + ' : ' + str(self.cycle_type),
               '  ' + '{0: <25}'.format(f'stopping tolerance') + ' : ' + str(self.tol),
               '  ' + '{0: <25}'.format(f'time communicator size') + ' : ' + str(self.comm_time_size),
               '  ' + '{0: <25}'.format(f'space communicator size') + ' : ' + str(self.comm_space_size),
               '  ' + '{0: <25}'.format(f'convergence criterion') + ' : ' + str(self.conv_crit)]
        self.log_info(message='\n'.join(msg))

    def solve(self) -> dict:
        """
        Driver function for solving the problem using MGRIT.

        Performs MGRIT iterations until a stopping criterion is fulfilled or
        the maximum number of iterations is reached.

        :return: dictionary with residual history, setup time, and solve time
        """
        # Start time of solve phase
        self.log_info("Start solve")

        runtime_solve_start = time.time()
        for iteration in range(self.iter_max):
            self.solve_iter = iteration + 1
            time_it_start = time.time()
            self.iteration(lvl=0, cycle_type=self.cycle_type, iteration=iteration, first_f=True)
            time_it_stop = time.time()
            self.convergence_criterion(iteration=iteration + 1)

            if iteration == 0:
                self.log_info('{0: <7}'.format(f"iter {iteration + 1}") +
                              ('{0: <32}'.format(f" | conv: {self.conv[iteration + 1]}") if self.global_conv_crit else
                              '{0: <32}'.format(
                                  f" | conv on process {self.comm_time_size - 1}: {self.conv[iteration + 1]}")) +
                              '{0: <37}'.format(f" | conv factor: -") +
                              '{0: <35}'.format(f" | runtime: {time_it_stop - time_it_start} s"))
            else:
                self.log_info('{0: <7}'.format(f"iter {iteration + 1}") +
                              ('{0: <32}'.format(f" | conv: {self.conv[iteration + 1]}") if self.global_conv_crit else
                              '{0: <32}'.format(
                                  f" | conv on process {self.comm_time_size - 1}: {self.conv[iteration + 1]}")) +
                              '{0: <37}'.format(f" | conv factor: {self.conv[iteration + 1] / self.conv[iteration]}") +
                              '{0: <35}'.format(f" | runtime: {time_it_stop - time_it_start} s"))

            if self.output_fcn is not None and self.output_lvl == 2:
                self.output_fcn(self)

            if self.conv[iteration + 1] < self.tol or iteration == self.iter_max - 1:
                if self.global_conv_crit:
                    self.clean_up()
                    break
                else:
                    if (self.finished[0] and self.pre_finished[0]) or iteration == self.iter_max - 1:
                        self.clean_up()
                        break
        # Stop timer of solve phase
        self.comm_time.barrier()
        self.runtime_solve = time.time() - runtime_solve_start
        self.log_info(f"Solve took {self.runtime_solve} s")

        if self.output_fcn is not None and self.output_lvl == 1:
            self.output_fcn(self)

        self.ouput_run_information()
        return {'conv': self.conv[np.where(self.conv != 0)], 'time_setup': self.runtime_setup,
                'time_solve': self.runtime_solve}

    def clean_up(self) -> None:
        MPI.Request.Waitall(self.requests)
        if not self.global_conv_crit:
            for lvl in range(self.lvl_max - 1):
                if self.last_is_c_point[lvl]:
                    self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=0)
                if self.comm_back[lvl]:
                    self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=1)
                if self.last_is_f_point[lvl]:
                    self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=2)
                if self.last_is_f_point[lvl]:
                    self.send(data=self.u[lvl][-1].pack(), dest=self.send_to[lvl], lvl=lvl, op_id=3)
                if self.send_to[lvl + 1] >= 0:
                    self.send(data=self.u[lvl][self.index_local_c[lvl][-1]].pack(), dest=self.send_to[lvl + 1],
                              lvl=lvl, op_id=4)
            if self.send_to[-1] != -99:
                self.send(data=self.u[-1][self.index_local[-1][-1]].pack(), dest=self.send_to[-1],
                          lvl=self.lvl_max - 1, op_id=5)

            # if self.comm_time_rank < self.comm_time_size - 1:
            #     self.send(data=self.finished[0], dest=self.send_to[0], lvl=0, op_id=6)
            if self.last_is_f_point[0]:
                self.send(data=self.u[0][-1].pack(), dest=self.comm_time_rank + 1, lvl=0, op_id=7)

            if self.finished[1] == self.pre_finished[1]:
                for lvl in range(self.lvl_max - 1):
                    if self.first_is_f_point[lvl] and not self.sender_finished[lvl][0]:
                        self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=0))
                    if self.comm_front[lvl] and not self.sender_finished[lvl][1]:
                        self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=1))
                    if self.first_is_c_point[lvl] and not self.sender_finished[lvl][2]:
                        self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=2))
                    if self.first_is_c_point[lvl] and not self.sender_finished[lvl][3]:
                        self.u[lvl][0].unpack(self.receive(source=self.get_from[lvl], lvl=lvl, op_id=3))
                    if self.get_from[lvl + 1] >= 0 and not self.sender_finished[lvl][4]:
                        tmp_last_c_point = self.problem[lvl].vector_template.clone_zero()
                        tmp_last_c_point.unpack(self.receive(source=self.get_from[lvl + 1], lvl=lvl, op_id=4))
                if self.get_from[-1] != -99 and not self.sender_finished[self.lvl_max - 1][5]:
                    self.u[-1][0].unpack(self.receive(source=self.get_from[-1], lvl=self.lvl_max - 1, op_id=5))
                # if self.comm_time_rank > 0 and not self.sender_finished[0][6]:
                #     tmp = self.receive(source=self.comm_time_rank - 1, lvl=0, op_id=6)
                if self.first_is_c_point[0] and not self.sender_finished[0][7]:
                    self.u[0][0].unpack(self.receive(source=self.get_from[0], lvl=0, op_id=7))
            MPI.Request.Waitall(self.requests)

    def receive(self, source, lvl, op_id):
        logging.debug(
            f"Comm: {self.comm_time_rank} recv from {source} |"
            f" lvl: {lvl} | op_id: {op_id} | tag: {self.tags_recv[lvl][op_id]}")
        tmp = self.comm_time.recv(source=source, tag=self.tags_recv[lvl][op_id])
        self.tags_recv[lvl][op_id] += 1
        if not self.global_conv_crit:
            self.sender_finished[lvl][op_id] = tmp[1]
            tmp = tmp[0]
        return tmp

    def send(self, data, dest, lvl, op_id):
        logging.debug(
            f"Comm: {self.comm_time_rank} send to {dest} |"
            f" lvl: {lvl} | op_id: {op_id} | tag: {self.tags_send[lvl][op_id]}")
        if self.global_conv_crit:
            self.requests.append(self.comm_time.isend(data, dest=dest, tag=self.tags_send[lvl][op_id]))
        else:
            self.requests.append(
                self.comm_time.isend([data, self.finished[0]], dest=dest, tag=self.tags_send[lvl][op_id]))
        self.tags_send[lvl][op_id] += 1

    def error_correction(self, lvl: int) -> None:
        """
        Computes the error approximation on level lvl and
        updates the approximation on the next finer level.

        :param lvl: MGRIT level
        """
        for i in range(len(self.index_local_c[lvl])):
            if i != 0 or self.comm_time_rank != 0:
                error = self.interpolation[lvl](
                    self.u[lvl + 1][self.index_local[lvl][i]] - self.v[lvl + 1][self.index_local[lvl][i]])
                self.u[lvl][self.index_local_c[lvl][i]] = self.u[lvl][self.index_local_c[lvl][i]] + error

    def split_points(self, length: int, size: int, rank: int) -> Tuple[int, int]:
        """
        Splits *length* points evenly in *size* parts and computes the index of the
        first point and block size of the local time interval.

        :param length: Number of points
        :param size: Number of processes
        :param rank: Process rank
        :return: Block size and index of first point
        """
        split = self.split_into(number_points=length, number_processes=size)

        return split[rank], np.sum(split[:rank]) if split[rank] > 0 else 0

    def setup_points_and_comm_info(self, lvl: int) -> None:
        """
        Computes local grid information for level *lvl*.
        Computes which process holds the previous and next point for each lvl and process.

        :param lvl: MGRIT level
        """
        self.global_t.append(np.copy(self.problem[lvl].t))
        points_time = np.size(self.global_t[lvl])
        all_pts = np.array(range(0, points_time))

        # Compute points per process
        # First level: Divide the points evenly
        # Other levels: Depends on the first level
        if lvl == 0:
            block_size_this_lvl, first_i_this_lvl = self.split_points(length=points_time,
                                                                      size=self.comm_time_size,
                                                                      rank=self.comm_time_rank)
            all_pts = all_pts[first_i_this_lvl:first_i_this_lvl + block_size_this_lvl]
            self.int_start = self.global_t[lvl][all_pts[0]]
            self.int_stop = self.global_t[lvl][all_pts[-1]]
        else:
            all_pts = np.where((self.global_t[lvl] >= self.int_start) & (self.global_t[lvl] <= self.int_stop))[0]

        # Compute C- and F-points
        if lvl != self.lvl_max - 1:
            all_cpts = np.where(np.in1d(self.problem[lvl].t, self.problem[lvl + 1].t))[0]
        else:
            all_cpts = np.array(range(0, points_time, self.m[lvl]))
        all_fpts = np.array(list(set(np.array(range(0, points_time))) - set(all_cpts)))
        cpts = np.sort(np.array(list(set(all_pts) - set(all_fpts)), dtype='int'))
        fpts = np.array(list(set(all_pts) - set(cpts)))
        fpts2 = np.array([item for sublist in np.array([np.array(xi, dtype=object) for xi in np.array(
            [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(fpts), lambda x: x[0] - x[1])], dtype=object)],
                                                       dtype=object)[::-1] for item in sublist])
        # Alternative
        # fpts3 = np.split(fpts, np.where(np.diff(fpts) != np.prod(self.m[:lvl]))[0] + 1)
        # fpts3 = np.array([item for sublist in fpts3[::-1] for item in sublist])

        # Add ghost point if needed, set time interval
        with_ghost_point = False
        if self.comm_time_rank != 0 and all_pts.size > 0:
            with_ghost_point = True
            tmp = np.zeros(len(all_pts) + 1, dtype=int)
            tmp[0] = all_pts[0] - 1
            tmp[1:] = all_pts
            all_pts_with_ghost = tmp
        else:
            all_pts_with_ghost = all_pts

        self.t[lvl] = self.global_t[lvl][all_pts_with_ghost]
        self.cpts.append(cpts)

        # Communication in F-relax
        self.comm_front.append(bool(fpts.size > 0 and fpts[np.argmin(fpts)] - 1 in all_fpts))
        self.comm_back.append(bool(fpts.size > 0 and fpts[np.argmax(fpts)] + 1 in all_fpts))

        # Setup local indices
        self.index_local.append(np.nonzero(all_pts[:, None] == all_pts_with_ghost)[1])
        self.index_local_c.append(np.nonzero(cpts[:, None] == all_pts_with_ghost)[1])
        self.index_local_f.append(np.nonzero(fpts2[:, None] == all_pts_with_ghost)[1])

        # Communication before and after C- and F-relax
        self.first_is_c_point.append(
            bool(all_pts.size > 0 and all_pts[0] in cpts and all_pts[0] != 0 and all_pts[0] - 1 in all_fpts))
        self.first_is_f_point.append(bool(all_pts.size > 0 and all_pts[0] in fpts2 and all_pts[0] - 1 in all_cpts))
        self.last_is_c_point.append(bool(
            all_pts.size > 0 and all_pts[-1] in cpts and all_pts[-1] != points_time - 1 and all_pts[
                -1] + 1 in all_fpts))
        self.last_is_f_point.append(bool(
            all_pts.size > 0 and all_pts[-1] in fpts2 and all_pts[-1] != points_time - 1 and all_pts[
                -1] + 1 in all_cpts))

        # Setup communication info
        split = self.global_t[0][
            np.cumsum(self.split_into(number_points=len(self.global_t[0]), number_processes=self.comm_time_size)) - 1]
        tmp_send_to = -99
        tmp_get_from = -99
        if len(all_pts_with_ghost) > 0:
            if self.t[lvl][-1] != self.global_t[lvl][-1]:
                last_point_next = self.global_t[lvl][np.argwhere(self.global_t[lvl] == self.t[lvl][-1])[0][0] + 1]
                tmp_send_to = np.searchsorted(split, last_point_next)
            if with_ghost_point or self.t[lvl][0] != self.global_t[0][0]:
                tmp_get_from = np.searchsorted(split, self.t[lvl][0])
        self.send_to.append(tmp_send_to)
        self.get_from.append(tmp_get_from)

    def split_into(self, number_points: int, number_processes: int) -> np.ndarray:
        """
        Split points

        :param number_points: Number of points
        :param number_processes: Number of processes
        :return:
        """
        return np.array([int(number_points / number_processes + 1)] * (number_points % number_processes) +
                        [int(number_points / number_processes)] * (number_processes - number_points % number_processes))

    def create_u_v_g(self, lvl: int) -> None:
        """
        Creates solution vectors for all local time points on a given MGRIT level.

        :param lvl: MGRIT level
        """
        if lvl == 0:
            if self.random_init_guess:
                self.u.append([self.problem[lvl].vector_template.clone_rand() for _ in range(len(self.t[lvl]))])
            else:
                self.u.append([self.problem[lvl].vector_template.clone_zero() for _ in range(len(self.t[lvl]))])
            self.v.append(None)
            self.g.append(None)
        else:
            self.u.append([self.problem[lvl].vector_template.clone_zero() for _ in range(len(self.t[lvl]))])
            self.v.append([item.clone_zero() for item in self.u[lvl]])
            self.g.append([item.clone_zero() for item in self.u[lvl]])
        if self.comm_time_rank == 0:
            self.u[lvl][0] = self.problem[lvl].vector_t_start.clone()
