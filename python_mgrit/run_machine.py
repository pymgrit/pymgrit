from mgrit import mgrit_fas as solver
from induction_machine import im_3kW
from induction_machine import grid_transfer_machine
from induction_machine import grid_transfer_copy
import numpy as np

if __name__ == '__main__':

    # machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_16k',
    #                                     t_start=0, t_stop=0.0002, nt=65)
    # first_level = np.hstack(
    #     (np.arange(0, len(machine_0.t))[::4], np.arange(0, len(machine_0.t))[::4][1:] - 1))
    # first_level.sort()
    # machine_0.t = machine_0.t[first_level]
    # machine_0.nt = len(first_level)
    # machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_16k',
    #                                     t_start=0, t_stop=0.0002, nt=17)
    # machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
    #                                     t_start=0, t_stop=0.0002, nt=5)
    #
    # problem = [machine_0, machine_1, machine_2]
    # transfer = [grid_transfer_copy.GridTransferCopy(),
    #             grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_16k', coarse_grid='im_3kW_4k')]
    # mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer)
    # result2 = mgrit.solve()

    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=0.0002, nt=2 ** 6 + 1)
    first_level = np.hstack(
        (np.arange(0, len(machine_0.t))[::4], np.arange(0, len(machine_0.t))[::4][1:] - 1))
    first_level.sort()
    machine_0.t = machine_0.t[first_level]
    machine_0.nt = len(first_level)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=0.0002, nt=2 ** 4 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=0.0002, nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer, it=1)
    result = mgrit.solve()

    1 + 1