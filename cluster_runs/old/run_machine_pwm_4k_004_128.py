from mgrit import mgrit_machine
from induction_machine import im_3kW
from induction_machine import grid_transfer_machine
from induction_machine import grid_transfer_copy
import logging

if __name__ == '__main__':
    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 7 + 1)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 3 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 1 + 1)

    problem = [machine_0, machine_1, machine_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = mgrit_machine.MgritMachine(compute_f_after_convergence=True, problem=problem,
                                       transfer=transfer, it=10, nested_iteration=True, tol=1,
                                       debug_lvl=logging.INFO)
    result_var = mgrit.solve()
