from mgrit import mgrit_machine as solver
from induction_machine import im_3kW
from induction_machine import grid_transfer_copy
import numpy as np
import pathlib

if __name__ == '__main__':
    def output_fcn(self):
        now = 'induction_machine_jl_sin_1'
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


    machine_0 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 15 + 1)
    machine_1 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 13 + 1)
    machine_2 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 11 + 1)
    machine_3 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 9 + 1)
    machine_4 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 7 + 1)
    machine_5 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 5 + 1)
    machine_6 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 3 + 1)
    machine_7 = im_3kW.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                        t_start=0, t_stop=2 ** -3, nt=2 ** 1 + 1)

    problem = [machine_0, machine_1, machine_2, machine_3, machine_4, machine_5, machine_6, machine_7]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritMachine(compute_f_after_convergence=True, problem=problem, transfer=transfer,
                                nested_iteration=True, output_fcn=output_fcn, tol=1)

    result = mgrit.solve()

    # u = result['u']
    # first_ua = []
    # second_ua = []
    # ua = []
    # first_ub = []
    # second_ub = []
    # ub = []
    # first_uc = []
    # second_uc = []
    # uc = []
    # stator_0 = []
    # stator_1 = []
    # stator_2 = []
    # stator_3 = []
    # stator_4 = []
    # stator_5 = []
    # stator_6 = []
    # stator_7 = []
    # ia = []
    # ib = []
    # ic = []
    # for i in range(len(u)):
    #     first_ua.append(u[i].u_back[3])
    #     second_ua.append(u[i].u_back[-3])
    #     ua.append(u[i].ua)
    #     first_ub.append(u[i].u_back[5])
    #     second_ub.append(u[i].u_back[-2])
    #     ub.append(u[i].ub)
    #     first_uc.append(u[i].u_back[4])
    #     second_uc.append(u[i].u_back[-1])
    #     uc.append(u[i].uc)
    #     stator_0.append(u[i].u_front[0])
    #     stator_1.append(u[i].u_front[1])
    #     stator_2.append(u[i].u_front[2])
    #     stator_3.append(u[i].u_front[3])
    #     stator_4.append(u[i].u_front[4])
    #     stator_5.append(u[i].u_front[5])
    #     stator_6.append(u[i].u_front[6])
    #     stator_7.append(u[i].u_front[7])
    #     ia.append(u[i].ia)
    #     ib.append(u[i].ib)
    #     ic.append(u[i].ic)
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.grid(True)
    # plt.plot(result['t'], first_ua, label='first')
    # plt.plot(result['t'], second_ua, label='second')
    # plt.plot(result['t'], ua, label='ua')
    # plt.legend()
    # np.savez('machine_back_ua_sin', first_ua=first_ua, second_ua=second_ua, ua=ua)
    # plt.savefig("machine_back_ua_sin.jpg", bbox_inches='tight')
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.grid(True)
    # plt.plot(result['t'], first_ub, label='first')
    # plt.plot(result['t'], second_ub, label='second')
    # plt.plot(result['t'], ub, label='uc')
    # plt.legend()
    # np.savez('machine_back_ub_sin', first_ub=first_ub, second_ub=second_ub, ub=ub)
    # plt.savefig("machine_back_ub_sin.jpg", bbox_inches='tight')
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.grid(True)
    # plt.plot(result['t'], first_uc, label='first')
    # plt.plot(result['t'], second_uc, label='second')
    # plt.plot(result['t'], uc, label='uc')
    # plt.legend()
    # np.savez('machine_back_uc_sin', first_uc=first_uc, second_uc=second_uc, uc=uc)
    # plt.savefig("machine_back_uc_sin.jpg", bbox_inches='tight')
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.grid(True)
    # plt.plot(result['t'], ia, label='ia')
    # plt.plot(result['t'], ib, label='ib')
    # plt.plot(result['t'], ic, label='ic')
    # plt.legend()
    # np.savez('machine_i_sin', ia=ia, ib=ib, ic=ic)
    # plt.savefig("machine_i_sin.jpg", bbox_inches='tight')
    # plt.show()
    #
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.grid(True)
    # plt.plot(result['t'], stator_0, label='stator 0')
    # plt.plot(result['t'], stator_1, label='stator 1')
    # plt.plot(result['t'], stator_2, label='stator 2')
    # plt.plot(result['t'], stator_3, label='stator 3')
    # plt.plot(result['t'], stator_4, label='stator 4')
    # plt.plot(result['t'], stator_5, label='stator 5')
    # plt.plot(result['t'], stator_6, label='stator 6')
    # plt.plot(result['t'], stator_7, label='stator 7')
    # plt.legend()
    # np.savez('stator_currents_sin', stator_0=stator_0, stator_1=stator_1, stator_2=stator_2, stator_3=stator_3,
    #         stator_4=stator_4, stator_5=stator_5, stator_6=stator_6, stator_7 = stator_7)
    # plt.savefig("stator_currents_sin.jpg", bbox_inches='tight')
    # plt.show()
