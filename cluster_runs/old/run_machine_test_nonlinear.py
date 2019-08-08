from mgrit import mgrit_machine
from induction_machine import im_3kW
from induction_machine import grid_transfer_machine
from induction_machine import grid_transfer_copy
import numpy as np
import logging
import pathlib

if __name__ == '__main__':
    def output_fcn(self):
        now = 'induction_machine_jl_4k_test'
        pathlib.Path('results/' + now).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'time': self.runtime_solve,
               'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + now + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)


    machine_0 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 13 + 1)
    machine_1 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 11 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 9 + 1)
    machine_3 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 7 + 1)
    machine_4 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 5 + 1)
    machine_5 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 3 + 1)
    machine_6 = im_3kW.InductionMachine(nonlinear=True, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 1 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4, machine_5, machine_6]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = mgrit_machine.MgritMachine(compute_f_after_convergence=True, problem=problem,
                                       transfer=transfer, it=10, nested_iteration=True, tol=1,
                                       debug_lvl=logging.INFO, output_fcn=output_fcn)
    result_var = mgrit.solve()
