from pymgrit.core import mgrit_machine
from pymgrit.induction_machine import im_3kW_non_uniform
from pymgrit.induction_machine import im_3kW
from pymgrit.induction_machine import grid_transfer_machine
from pymgrit.induction_machine import grid_transfer_copy
import numpy as np
import logging
import pathlib
from matplotlib import pyplot as plt

def signal_pwm(t):
    freq = 50
    teeth = 400
    freq_pwm = freq * teeth
    FSaw_fc = 2 * (t * freq_pwm - np.floor(t * freq_pwm)) - 1
    FCarrier = FSaw_fc

    phase_m_A = 0
    phase_m_B = -2 / 3 * np.pi
    phase_m_C = -4 / 3 * np.pi

    FV_out_A = np.sin((2 * np.pi * freq * t) + phase_m_A)
    FV_out_B = np.sin((2 * np.pi * freq * t) + phase_m_B)
    FV_out_C = np.sin((2 * np.pi * freq * t) + phase_m_C)

    F_PWM_A = np.sign(FV_out_A - FCarrier)
    F_PWM_B = np.sign(FV_out_B - FCarrier)
    F_PWM_C = np.sign(FV_out_C - FCarrier)

    #frelax = 0.5* (1-np.cos(np.pi*t/(2*0.02)))
    frelax = 1
    voltage = 1
    pwm1 = frelax * F_PWM_A * voltage
    pwm2 = frelax * F_PWM_B * voltage
    pwm3 = frelax * F_PWM_C * voltage

    #period_max = np.where(t<=0.02)[0]
    # diff1 = pwm1[period_max][1:] - pwm1[period_max][:-1]
    # diff2 = pwm2[period_max][1:] - pwm2[period_max][:-1]
    # diff3 = pwm3[period_max][1:] - pwm3[period_max][:-1]

    diff1 = pwm1[1:] - pwm1[:-1]
    diff2 = pwm2[1:] - pwm2[:-1]
    diff3 = pwm3[1:] - pwm3[:-1]

    print('Machine pwm with', len(t), 'timepoints and pwm frequency', freq_pwm, 'results in (',
          len(np.where(diff1 != 0)[0]), len(np.where(diff2 != 0)[0]), len(np.where(diff3 != 0)[0]),
          ')changes for the three phases')
    plt.plot(t,FCarrier, 'x')
    plt.plot(t,FV_out_A,'s')
    plt.plot(t,pwm1, '.')
    plt.plot(t,pwm2, '.')
    plt.plot(t,pwm3, '.')
    plt.show()

if __name__ == '__main__':
    def output_fcn(self):
        now = 'induction_machine_non_uniform'
        pathlib.Path('results/' + now).mkdir(parents=True, exist_ok=True)
        jl = [self.u[0][i].jl for i in self.index_local[0]]
        ia = [self.u[0][i].ia for i in self.index_local[0]]
        ib = [self.u[0][i].ib for i in self.index_local[0]]
        ic = [self.u[0][i].ic for i in self.index_local[0]]
        ua = [self.u[0][i].ua for i in self.index_local[0]]
        ub = [self.u[0][i].ub for i in self.index_local[0]]
        uc = [self.u[0][i].uc for i in self.index_local[0]]
        last_0 = 0
        last_1 = 0
        last_2 = 0
        if self.comm_time_rank == self.comm_time_size - 1:
            last_0 = self.u[0][-1]
            last_1 = self.u[1][-1]
            last_2 = self.u[2][-1]
        sol = {'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'time': self.runtime_solve,
               'conv': self.conv, 't': self.problem[0].t, 'time_setup': self.runtime_setup, 'last_0': last_0,
               'last_1': last_1, 'last_2': last_2}

        np.save('results/' + now + '/' + str(self.t[0][-1]), sol)


    # first_t = np.linspace(0, 0.02, 6401)
    # small_2er = 2 ** -24
    # t_neu = [first_t[0]]
    # ind = np.where(np.abs(first_t - 5e-05) <= 0.0000000001)[0]
    # for i in range(1, len(first_t)):
    #     if i % ind == 0:
    #         lower = np.floor(first_t[i] / small_2er)
    #         upper = np.ceil(first_t[i] / small_2er)
    #         t_neu.append(lower * small_2er)
    #         t_neu.append(upper * small_2er)
    #     else:
    #         lower = np.floor(first_t[i] / small_2er)
    #         upper = np.ceil(first_t[i] / small_2er)
    #         if np.abs(np.abs(lower * small_2er) - np.abs(first_t[i])) <= np.abs(
    #                 np.abs(upper * small_2er) - np.abs(first_t[i])):
    #             t_neu.append(lower * small_2er)
    #         else:
    #             t_neu.append(upper * small_2er)
    # first_t = np.array(t_neu)
    # second_t = np.array(first_t[::16])
    # third_t = np.array(second_t[::5])
    # fourth_t = np.array(third_t[::17])

    first_t = np.linspace(0, 0.02, 12801)
    small_2er = 2 ** -24
    t_neu = [first_t[0]]
    ind = np.where(np.abs(first_t - 5e-05) <= 0.0000000001)[0]
    for i in range(1, len(first_t)):
        if i % ind == 0:
            lower = np.floor(first_t[i] / small_2er)
            upper = np.ceil(first_t[i] / small_2er)
            t_neu.append(lower * small_2er)
            t_neu.append(upper * small_2er)
        else:
            lower = np.floor(first_t[i] / small_2er)
            upper = np.ceil(first_t[i] / small_2er)
            if np.abs(np.abs(lower * small_2er) - np.abs(first_t[i])) <= np.abs(
                    np.abs(upper * small_2er) - np.abs(first_t[i])):
                t_neu.append(lower * small_2er)
            else:
                t_neu.append(upper * small_2er)
    first_t = np.array(t_neu)
    second_t = np.array(first_t[::50])
    third_t = np.array(second_t[::4])
    fourth_t = np.array(third_t[::4])
    fives_t = np.array(fourth_t[::4])

    t_old_1 = np.linspace(0,2**-5,2**15+1)
    t_old_2 = np.linspace(0,2**-5,2**10+1)
    t_old_3 = np.linspace(0,2**-5,2**7+1)
    t_old_4 = np.linspace(0,2**-5,2**3+1)

    signal_pwm(t_old_1)
    signal_pwm(t_old_2)
    signal_pwm(t_old_3)
    signal_pwm(t_old_4)

    t_old_1 = np.linspace(0,2**-5,20001)
    t_old_2 = t_old_1[::50]

    signal_pwm(t_old_1)
    signal_pwm(t_old_2)
    #signal_pwm(t_old_3)
    #signal_pwm(t_old_4)

    signal_pwm(first_t)
    signal_pwm(second_t)
    signal_pwm(third_t)
    signal_pwm(fourth_t)
    signal_pwm(fives_t)

    # Complete
    machine_0 = im_3kW_non_uniform.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                                    t_start=0, t_stop=2 ** -7, nt=2 ** 9 + 1, t=first_t)
    machine_1 = im_3kW_non_uniform.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                                    t_start=0, t_stop=2 ** -7, nt=2 ** 5 + 1, t=second_t)
    machine_2 = im_3kW_non_uniform.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                                    t_start=0, t_stop=2 ** -7, nt=2 ** 1 + 1, t=third_t)
    machine_3 = im_3kW_non_uniform.InductionMachine(nonlinear=False, pwm=False, grid='im_3kW_4k',
                                                    t_start=0, t_stop=2 ** -7, nt=2 ** 1 + 1, t=fourth_t)

    problem = [machine_0, machine_1, machine_2, machine_3]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy(),
                grid_transfer_copy.GridTransferCopy()]
    mgrit = mgrit_machine.MgritMachine(compute_f_after_convergence=True, problem=problem,
                                       transfer=transfer, it=10, nested_iteration=True, tol=1,
                                       output_fcn=output_fcn)
    result_var_complete = mgrit.solve()
