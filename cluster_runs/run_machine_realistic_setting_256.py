from mgrit import mgrit_machine as solver
from induction_machine import im_3kW
from induction_machine import grid_transfer_copy
from induction_machine import grid_transfer_machine
import numpy as np
import pathlib

if __name__ == '__main__':
    def output_fcn(self):
        now = 'ind_mac_realistic_256'
        pathlib.Path('results/' + now).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        back_3 = [self.u[0][i].u_back[3] for i in self.index_local[0]]
        back_4 = [self.u[0][i].u_back[4] for i in self.index_local[0]]
        back_5 = [self.u[0][i].u_back[5] for i in self.index_local[0]]
        back_12 = [self.u[0][i].u_back[12] for i in self.index_local[0]]
        back_13 = [self.u[0][i].u_back[13] for i in self.index_local[0]]
        back_14 = [self.u[0][i].u_back[14] for i in self.index_local[0]]
        front0 = [self.u[0][i].u_front[0] for i in self.index_local[0]]
        front1 = [self.u[0][i].u_front[1] for i in self.index_local[0]]
        front2 = [self.u[0][i].u_front[2] for i in self.index_local[0]]
        front3 = [self.u[0][i].u_front[3] for i in self.index_local[0]]
        front4 = [self.u[0][i].u_front[4] for i in self.index_local[0]]
        front5 = [self.u[0][i].u_front[5] for i in self.index_local[0]]
        front6 = [self.u[0][i].u_front[6] for i in self.index_local[0]]
        front7 = [self.u[0][i].u_front[7] for i in self.index_local[0]]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'time': self.runtime_solve,
               'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup, 'back3': back_3,
               'back4': back_4, 'back5': back_5, 'back12': back_12, 'back13': back_13, 'back14': back_14,
               'front0': front0, 'front1': front1, 'front2': front2, 'front3': front3, 'front4': front4,
               'front5': front5, 'front6': front6, 'front7': front7}

        np.save('results/' + now + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)


    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_69k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 20 + 1)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_17k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 16 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 12 + 1)
    machine_3 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 10 + 1)
    machine_4 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 8 + 1)
    machine_5 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 6 + 1)
    machine_6 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 4 + 1)
    machine_7 = im_3kW.InductionMachine(nonlinear=False, pwm=True, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -5, nt=2 ** 2 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4, machine_5, machine_6, machine_7]
    transfer = [grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_69k', coarse_grid='im_3kW_17k'),
                grid_transfer_machine.GridTransferMachine(fine_grid='im_3kW_17k', coarse_grid='im_3kW_4k'),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritMachine(compute_f_after_convergence=True, problem=problem, transfer=transfer,
                                nested_iteration=True, tol=1, output_fcn=output_fcn)

    res = mgrit.solve()
