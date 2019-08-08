from mgrit import mgrit_machine
from induction_machine import im_3kW
from induction_machine import grid_transfer_machine
from induction_machine import grid_transfer_copy
import numpy as np
import logging
import pathlib

if __name__ == '__main__':
    def output_fcn(self):
        now = 'induction_machine_jl_69k'
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

    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_69k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 13 + 1)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_17k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 7 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -8, nt=2 ** 1 + 1)

    problem = [machine_0, machine_1, machine_2]
    transfer = [grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_69k', coarse_grid='im_3kW_17k'),
                grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_17k', coarse_grid='im_3kW_4k')]
    mgrit = mgrit_machine.MgritMachine(compute_f_after_convergence=True, problem=problem,
                                       transfer=transfer, it=10, nested_iteration=True, tol=1,
                                       debug_lvl=logging.INFO, output_fcn=output_fcn)
    result_var = mgrit.solve()
