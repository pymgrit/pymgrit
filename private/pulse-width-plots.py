import numpy as np
import matplotlib.pyplot as plt


def pwm_classic_cable(t_end, nt, freq, number_tooth):
    t = np.linspace(0, t_end, nt)
    freq_pwm = freq * number_tooth
    saw = t * freq_pwm - np.floor(t * freq_pwm)

    sine = np.sin(freq * t * (2 * np.pi))

    case = (saw - abs(sine) < 0)

    diff = case[:-1].astype(int) - case[1:].astype(int)

    print('Cable pwm with', nt, 'timepoints and pwm frequency', freq_pwm, 'results in', len(np.where(diff != 0)[0]),
          'changes')
    # plt.plot(t,saw)
    # plt.plot(t,sine)
    # plt.plot(t,case)
    # plt.show()


def pwm_machine(t_end, nt, freq, number_tooth):
    t = np.linspace(0, t_end, nt)
    freq_pwm = freq * number_tooth
    saw = t * freq_pwm - np.floor(t * freq_pwm)

    FSaw_fc = 2 * (t * freq_pwm - np.floor(t * freq_pwm)) - 1
    FCarrier = FSaw_fc

    phase_m_A = 0
    phase_m_B = -2 / 3 * np.pi
    phase_m_C = -4 / 3 * np.pi

    FV_out_A = np.sin((2 * np.pi * freq * t) + phase_m_A)
    FV_out_B = np.sin((2 * np.pi * freq * t) + phase_m_B)
    FV_out_C = np.sin((2 * np.pi * freq * t) + phase_m_C)

    # plt.plot(t, FCarrier)
    # plt.plot(t, FV_out_A, color='red')
    # plt.plot(t, FV_out_B, color = 'green')
    # plt.plot(t, FV_out_C, color = 'orange')
    # plt.show()
    F_PWM_A = np.sign(FV_out_A - FCarrier)
    F_PWM_B = np.sign(FV_out_B - FCarrier)
    F_PWM_C = np.sign(FV_out_C - FCarrier)

    frelax = 0.5* (1-np.cos(np.pi*t/(2*0.02)))
    #frelax = 1
    voltage = 1
    pwm1 = frelax * F_PWM_A * voltage
    pwm2 = frelax * F_PWM_B * voltage
    pwm3 = frelax * F_PWM_C * voltage

    diff1 = pwm1[1:] - pwm1[:-1]
    diff2 = pwm2[1:] - pwm2[:-1]
    diff3 = pwm3[1:] - pwm3[:-1]
    print('Machine pwm with', nt, 'timepoints and pwm frequency', freq_pwm, 'results in (',
          len(np.where(diff1 != 0)[0]), len(np.where(diff2 != 0)[0]), len(np.where(diff3 != 0)[0]),
          ')changes for the three phases')
    plt.plot(t,pwm1)
    plt.plot(t,pwm2)
    plt.plot(t,pwm3)
    plt.show()

def pwm_machine_2(t_end, nt, freq, number_tooth):
    dt = t_end / (nt-1)
    freq_pwm = freq * number_tooth

    phase_m_A = 0
    phase_m_B = -2 / 3 * np.pi
    phase_m_C = -4 / 3 * np.pi

    def computeValuePhaseA(t):
        FSaw_fc = 2 * (t * freq_pwm - np.floor(t * freq_pwm)) - 1
        FCarrier = FSaw_fc
        FV_out_A = np.sin((2 * np.pi * freq * t) + phase_m_A)
        F_PWM_A = np.sign(FV_out_A - FCarrier)
        frelax = 1
        voltage = 1
        pwm1 = frelax * F_PWM_A * voltage
        return pwm1

    def computeValuePhaseB(t):
        FSaw_fc = 2 * (t * freq_pwm - np.floor(t * freq_pwm)) - 1
        FCarrier = FSaw_fc
        FV_out_B = np.sin((2 * np.pi * freq * t) + phase_m_B)
        F_PWM_B = np.sign(FV_out_B - FCarrier)
        frelax = 1
        voltage = 1
        pwm2 = frelax * F_PWM_B * voltage
        return pwm2

    def computeValuePhaseC(t):
        FSaw_fc = 2 * (t * freq_pwm - np.floor(t * freq_pwm)) - 1
        FCarrier = FSaw_fc
        FV_out_C = np.sin((2 * np.pi * freq * t) + phase_m_C)
        F_PWM_C = np.sign(FV_out_C - FCarrier)
        frelax = 1
        voltage = 1
        pwm3 = frelax * F_PWM_C * voltage
        return pwm3

    lastA = computeValuePhaseA(0)
    lastB = computeValuePhaseB(0)
    lastC = computeValuePhaseC(0)

    diff1 = 0
    diff2 = 0
    diff3 = 0
    for i in range(1,nt):
        if (computeValuePhaseA(i*dt) - lastA) != 0:
            lastA = computeValuePhaseA(i*dt)
            diff1 += 1
        if (computeValuePhaseB(i*dt) - lastB) != 0:
            lastB = computeValuePhaseB(i*dt)
            diff2 += 1
        if (computeValuePhaseC(i*dt) - lastC) != 0:
            lastC = computeValuePhaseC(i*dt)
            diff3 += 1

    print('Machine pwm with', nt, 'timepoints and pwm frequency', freq_pwm, 'results in (',
          diff1, diff2, diff3,')changes for the three phases')


tooth =400
freq = 50
stop = 0.02
#
# pwm_classic_cable(0.02, 10001, freq, tooth)
# pwm_classic_cable(0.02, 20001, freq, tooth)
# pwm_classic_cable(0.02, 40001, freq, tooth)
# pwm_classic_cable(0.02, 80001, freq, tooth)
# pwm_classic_cable(0.02, 160001, freq, tooth)
# pwm_classic_cable(0.02, 320001, freq, tooth)
# pwm_classic_cable(0.02, 640001, freq, tooth)
# pwm_classic_cable(0.02, 1280001, freq, tooth)
#pwm_machine(0.02, 10001, freq, tooth)
# pwm_machine(0.02, 20001, freq, tooth)
# pwm_machine(0.02, 40001, freq, tooth)
# pwm_machine(0.02, 80001, freq, tooth)
# pwm_machine(0.02, 160001, freq, tooth)
# pwm_machine(0.02, 320001, freq, tooth)
pwm_machine(0.02, 640001, freq, tooth)
# pwm_machine(0.02, 1280001, freq, tooth)
#pwm_machine_2(0.02, 320001, freq, tooth)
#pwm_machine_2(0.02, 640001, freq, tooth)
# pwm_machine_2(stop, 2**9+1, freq, tooth)
# pwm_machine_2(stop, 2**10+1, freq, tooth)
# pwm_machine_2(stop, 2**11+1, freq, tooth)
# pwm_machine_2(stop, 2**12+1, freq, tooth)
# pwm_machine_2(stop, 2**13+1, freq, tooth)
# pwm_machine_2(stop, 2**14+1, freq, tooth)
# pwm_machine_2(stop, 2**15+1, freq, tooth)
# pwm_machine_2(stop, 2**16+1, freq, tooth)
# pwm_machine_2(stop, 2**17+1, freq, tooth)
# pwm_machine_2(stop, 2**18+1, freq, tooth)
# pwm_machine_2(stop, 2**19+1, freq, tooth)
# pwm_machine_2(stop, 2**20+1, freq, tooth)
# pwm_machine_2(stop, 2**21+1, freq, tooth)
# pwm_machine_2(stop, 2**22+1, freq, tooth)
# pwm_machine_2(stop, 2**23+1, freq, tooth)
# pwm_machine_2(stop, 2**24+1, freq, tooth)
# pwm_machine_2(stop, 2**25+1, freq, tooth)
# pwm_machine_2(stop, 2**26+1, freq, tooth)
# pwm_machine_2(stop, 2**27+1, freq, tooth)
# pwm_machine_2(stop, 2**28+1, freq, tooth)
# pwm_machine_2(stop, 2**29+1, freq, tooth)
