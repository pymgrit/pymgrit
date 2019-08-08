from mpi4py import MPI
from mgrit import mgrit_machine
from induction_machine import im_3kW
from induction_machine import grid_transfer_machine
import numpy as np
import logging


if __name__ == '__main__':
    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_64k',
                                        t_start=0, t_stop=2**-3, nt=2**14+1)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_16k',
                                        t_start=0, t_stop=2**-3, nt=2**9+1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2**-3, nt=2**5+1)

    problem = [machine_0, machine_1, machine_2]
    transfer = [grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_64k', coarse_grid='im_3kW_16k'),
                grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_16k', coarse_grid='im_3kW_4k')]
    mgrit = mgrit_machine.MgritMachine(compute_f_after_convergence=True, problem=problem,
                                       transfer=transfer, it=10, nested_iteration=True, tol=1,
                                       debug_lvl=logging.INFO)
    result_var = mgrit.solve()
