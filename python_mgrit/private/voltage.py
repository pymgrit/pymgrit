import numpy as np
import matplotlib.pyplot as plt


def plotVoltageOriginal(m=20):
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    t = np.linspace(0, 0.02, 200)
    w = np.sin((2 * np.pi / 0.02) * t)
    modulationFactor = 0.8
    voltage = 220 * np.sqrt(2)
    q2 = np.sign(
        (modulationFactor * np.sin(2 * np.pi * 50 + 0)) - (2 * (t * (m / 0.02) - np.floor(t * (m / 0.02))) - 1))
    q3 = np.sign((modulationFactor * np.sin(2 * np.pi * 50 + (-2 / 3 * np.pi))) - (
                2 * (t * (m / 0.02) - np.floor(t * (m / 0.02))) - 1))
    q4 = np.sign((modulationFactor * np.sin(2 * np.pi * 50 + (-4 / 3 * np.pi))) - (
                2 * (t * (m / 0.02) - np.floor(t * (m / 0.02))) - 1))
    frelax = 0.5 * (1 - np.cos(np.pi * t / (2 * 0.02)))
    pwm = frelax * q2 * voltage
    pwm2 = frelax * q3 * voltage
    pwm3 = frelax * q4 * voltage
    sin = frelax * w * voltage * modulationFactor
    plt.grid(True)
    plt.plot(t, pwm, color='red')
    # plt.plot(t,pwm2)
    # plt.plot(t,pwm3)
    plt.plot(t, sin, color='blue')
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.xlabel('Time / s')
    plt.ylabel('Voltage / V')
    plt.savefig("voltage.eps", bbox_inches='tight')
    plt.show()


def plotVoltage(Tend, steps):
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    t = np.linspace(0, Tend, steps)
    w = np.sin((2 * np.pi / 0.02) * t)
    modulationFactor = 1
    voltage = 220 * np.sqrt(2)

    Freq = 50
    FreqPWM = 20000
    Ac = 1
    Am = 1 * modulationFactor

    phase_m_A = 0
    phase_m_B = -2 / 3 * np.pi
    phase_m_C = -4 / 3 * np.pi
    fm = Freq
    fc = FreqPWM

    FSaw_fc = 2 * (t * fc - np.floor(t * fc)) - 1
    FCarrier = Ac * FSaw_fc

    FV_out_A = Am * np.sin((2 * np.pi * Freq * t) + phase_m_A)
    FV_out_B = Am * np.sin((2 * np.pi * Freq * t) + phase_m_B)
    FV_out_C = Am * np.sin((2 * np.pi * Freq * t) + phase_m_C)

    plt.plot(t, FCarrier)
    plt.plot(t, FV_out_A, color='red')
    plt.plot(t, FV_out_B, color = 'green')
    plt.plot(t, FV_out_C, color = 'orange')
    plt.show()
    F_PWM_A = np.sign(FV_out_A - FCarrier)
    F_PWM_B = np.sign(FV_out_B - FCarrier)
    F_PWM_C = np.sign(FV_out_C - FCarrier)

    # frelax = 0.5* (1-np.cos(np.pi*t/(2*0.02)))
    frelax = 1
    voltage = 1
    pwm = frelax * F_PWM_A * voltage
    pwm2 = frelax * F_PWM_B * voltage
    pwm3 = frelax * F_PWM_C * voltage

    diff = pwm[1:] - pwm[:-1]
    diff2 = pwm2[1:] - pwm2[:-1]
    diff3 = pwm3[1:] - pwm3[:-1]
    print('end', Tend, '| steps', steps, '|', np.where(diff != 0)[0].size, '|', np.where(diff2 != 0)[0].size, '|',
          np.where(diff3 != 0)[0].size, '| dt', t[1])
    sin = frelax * w * voltage * modulationFactor
    plt.grid(True)
    # plt.plot(t,pwm, color = 'red')
    # plt.plot(t,pwm2, color = 'green')
    # plt.plot(t,pwm3, color = 'orange')
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.xlabel('Time / s')
    plt.ylabel('Voltage / V')
    plt.savefig("voltage.eps", bbox_inches='tight')
    # plt.show()


def utlPwm(t, freq, teeth):
    # sawfish pattern with higher frequency
    saw = t * teeth * freq - np.floor(t * teeth * freq)

    # plain sine wave
    sine = np.sin(freq * t * (2 * np.pi))

    # pwm signal by comparison
    pwm = np.sign(sine) * (saw - abs(sine) < 0)

    return pwm


def plotPwm():
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)

    t = np.linspace(0, 0.02, 20001)
    pwm = utlPwm(t, 50, teeth=400)
    sin = np.sin(50 * t * 2 * np.pi)
    diff = pwm[1:] - pwm[:-1]
    print(np.where(diff != 0)[0].size)
    plt.plot(t, pwm, color='red')
    plt.xlabel('Time / s')
    plt.ylabel('Input function')
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    # plt.savefig("pwm1.eps", bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('Time / s')
    plt.ylabel('Input function')
    plt.plot(t, pwm, color='red')
    plt.plot(t, sin, color='blue')
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    # plt.savefig("pwm2.eps", bbox_inches='tight')
    # plt.show()


# plotPwm()
#plotVoltage(2 ** -8, 2**14+1)
plotVoltage(0.02, 20000)
plotVoltage(0.02, 200000)
plotVoltage(0.02, 2000000)
# plotVoltage(2 ** -20 * 4096 * 4, 4096 )
# plotVoltage(2 ** -20 * 4096 * 4, 4096 * 2)
# plotVoltage(2 ** -20 * 4096 * 4, 4096 * 4)
# plotVoltage(2 ** -20 * 4096 * 4, 4096 * 8)
