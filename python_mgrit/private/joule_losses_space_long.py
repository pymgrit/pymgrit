import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from operator import itemgetter


def plot_solution(t, sol, label):
    jl = np.zeros(len(sol))

    for i in range(len(sol)):
        jl[i] = sol[i].jl

    return jl


def solution(filelist):
    sol = []
    for infile in sorted(filelist):
        a = np.load(infile, allow_pickle=True).item()
        sol.append(a['u'])
        t = a['t']
        runtime = a['time']
    sol = [item for sublist in sol for item in sublist]
    return sol, t, runtime


# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-4k', '*'))
# sol, t_4k_sin, runtime_4k_sin = solution(filelist)
# jl_4k_sin = jl_solution(t_4k_sin, sol, 'Sinus signal')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-4k', '*'))
# sol, t_4k_pwm, runtime_4k_pwm = solution(filelist)
# jl_4k_pwm = jl_solution(t_4k_pwm, sol, 'PWM signal')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-16k', '*'))
# sol, t_16k_sin, runtime_16k_sin = solution(filelist)
# jl_16k_sin = jl_solution(t_16k_sin, sol, 'Sinus signal')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-16k', '*'))
# sol, t_16k_pwm, runtime_16k_pwm = solution(filelist)
# jl_16k_pwm = jl_solution(t_16k_pwm, sol, 'PWM signal')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-64k', '*'))
# sol, t_64k_sin, runtime_64k_sin = solution(filelist)
# jl_64k_sin = jl_solution(t_64k_sin, sol, 'Sinus signal')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-64k', '*'))
# sol, t_64k_pwm, runtime_64k_pwm = solution(filelist)
# jl_64k_pwm = jl_solution(t_64k_pwm, sol, 'PWM signal')
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-sin-4k-compact', jl_4k_sin=jl_4k_sin, t_4k_sin=t_4k_sin,
#          runtime_4k_sin=runtime_4k_sin)
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-pwm-4k-compact', jl_4k_pwm=jl_4k_pwm, t_4k_pwm=t_4k_pwm,
#          runtime_4k_pwm=runtime_4k_pwm)
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-sin-16k-compact', jl_16k_sin=jl_16k_sin, t_16k_sin=t_16k_sin,
#          runtime_16k_sin=runtime_16k_sin)
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-pwm-16k-compact', jl_16k_pwm=jl_16k_pwm, t_16k_pwm=t_16k_pwm,
#          runtime_16k_pwm=runtime_16k_pwm)
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-sin-64k-compact', jl_64k_sin=jl_64k_sin, t_64k_sin=t_64k_sin,
#          runtime_64k_sin=runtime_64k_sin)
#
# np.savez('/home/jens/uni/results/t0-0.125-16385-pwm-64k-compact', jl_64k_pwm=jl_64k_pwm, t_64k_pwm=t_64k_pwm,
#          runtime_64k_pwm=runtime_64k_pwm)

sin_4k = np.load('/home/jens/uni/results/t0-0.125-16385-sin-4k-compact.npz')
jl_4k_sin = sin_4k['jl_4k_sin']
t_4k_sin = sin_4k['t_4k_sin']
runtime_4k_sin = sin_4k['runtime_4k_sin']

pwm_4k = np.load('/home/jens/uni/results/t0-0.125-16385-pwm-4k-compact.npz')
jl_4k_pwm = pwm_4k['jl_4k_pwm']
t_4k_pwm = pwm_4k['t_4k_pwm']
runtime_4k_pwm = pwm_4k['runtime_4k_pwm']

sin_16k = np.load('/home/jens/uni/results/t0-0.125-16385-sin-16k-compact.npz')
jl_16k_sin = sin_16k['jl_16k_sin']
t_16k_sin = sin_16k['t_16k_sin']
runtime_16k_sin = sin_16k['runtime_16k_sin']

pwm_16k = np.load('/home/jens/uni/results/t0-0.125-16385-pwm-16k-compact.npz')
jl_16k_pwm = pwm_16k['jl_16k_pwm']
t_16k_pwm = pwm_16k['t_16k_pwm']
runtime_16k_pwm = pwm_16k['runtime_16k_pwm']

sin_64k = np.load('/home/jens/uni/results/t0-0.125-16385-sin-64k-compact.npz')
jl_64k_sin = sin_64k['jl_64k_sin']
t_64k_sin = sin_64k['t_64k_sin']
runtime_64k_sin = sin_64k['runtime_64k_sin']

pwm_64k = np.load('/home/jens/uni/results/t0-0.125-16385-pwm-64k-compact.npz')
jl_64k_pwm = pwm_64k['jl_64k_pwm']
t_64k_pwm = pwm_64k['t_64k_pwm']
runtime_64k_pwm = pwm_64k['runtime_64k_pwm']

print('runtime sin', runtime_4k_sin, runtime_16k_sin, runtime_64k_sin)
print('runtime pwm', runtime_4k_pwm, runtime_16k_pwm, runtime_64k_pwm)

diff_sin_64_16 = np.abs(jl_64k_sin[1:] - jl_16k_sin[1:])
diff_pwm_64_16 = np.abs(jl_64k_pwm[1:] - jl_16k_pwm[1:])

diff_sin_64_4 = np.abs(jl_64k_sin[1:] - jl_4k_sin[1:])
diff_pwm_64_4 = np.abs(jl_64k_pwm[1:] - jl_4k_pwm[1:])

fonts = 18

per_1 = np.where((t_4k_pwm > 0) & (t_4k_pwm <= 0.02))
per_2 = np.where((t_4k_pwm > 0.02) & (t_4k_pwm <= 0.04))
per_3 = np.where((t_4k_pwm > 0.04) & (t_4k_pwm <= 0.06))
per_4 = np.where((t_4k_pwm > 0.06) & (t_4k_pwm <= 0.08))
per_5 = np.where((t_4k_pwm > 0.08) & (t_4k_pwm <= 0.1))
per_6 = np.where((t_4k_pwm > 0.1) & (t_4k_pwm <= 0.12))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
plt.plot(t_4k_pwm[1:], diff_pwm_64_4, label='Absolute difference pwm 69k vs 4k', lw=2)
plt.plot(t_4k_pwm[1:], diff_pwm_64_16, label='Absolute difference pwm 69k vs 17k', lw=2)
print('mean absolute pwm 67 vs 4', np.mean(diff_pwm_64_4))
print('mean absolute pwm 67 vs 17', np.mean(diff_pwm_64_16))

print('max absolute pwm 67 vs 4', np.max(diff_pwm_64_4))
print('max absolute pwm 67 vs 17', np.max(diff_pwm_64_16))

print('mean absolute period 1 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_1]))
print('mean absolute period 2 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_2]))
print('mean absolute period 3 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_3]))
print('mean absolute period 4 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_4]))
print('mean absolute period 5 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_5]))
print('mean absolute period 6 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_6]))

print('mean absolute period 1 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_1]))
print('mean absolute period 2 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_2]))
print('mean absolute period 3 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_3]))
print('mean absolute period 4 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_4]))
print('mean absolute period 5 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_5]))
print('mean absolute period 6 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_6]))

plt.yscale('log')
plt.xlabel('Time [s]', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower left', prop={'size': fonts})
plt.savefig("joule_losses_space_absolute.jpg", bbox_inches='tight')
plt.show()


print('-----------------------------------------')
# plt.plot(t_4k_pwm[1:], diff_sin_64_4, label='Absolute error sin 67 vs 4')
# plt.plot(t_4k_pwm[1:], diff_sin_64_16, label='Absolute error sin 67 vs 17')
# print('mean absolute sin 67 vs 4', np.mean(diff_sin_64_4))
# print('mean absolute sin 67 vs 17', np.mean(diff_sin_64_16))
#
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()

diff_sin_64_16 = np.abs(jl_64k_sin[1:] - jl_16k_sin[1:]) / jl_64k_sin[1:]
diff_pwm_64_16 = np.abs(jl_64k_pwm[1:] - jl_16k_pwm[1:]) / jl_64k_pwm[1:]

diff_sin_64_4 = np.abs(jl_64k_sin[1:] - jl_4k_sin[1:]) / jl_64k_sin[1:]
diff_pwm_64_4 = np.abs(jl_64k_pwm[1:] - jl_4k_pwm[1:]) / jl_64k_pwm[1:]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
plt.plot(t_4k_pwm[1:], diff_pwm_64_4, label='Relative difference pwm 69k vs 4k', lw=2)
plt.plot(t_4k_pwm[1:], diff_pwm_64_16, label='Relative difference pwm 69k vs 17k', lw=2)
print('mean relativ pwm 67 vs 4', np.mean(diff_pwm_64_4))
print('mean relativ pwm 67 vs 17', np.mean(diff_pwm_64_16))
print('max relativ pwm 67 vs 4', np.max(diff_pwm_64_4))
print('max relativ pwm 67 vs 17', np.max(diff_pwm_64_16))

print('mean relative period 1 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_1]))
print('mean relative period 2 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_2]))
print('mean relative period 3 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_3]))
print('mean relative period 4 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_4]))
print('mean relative period 5 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_5]))
print('mean relative period 6 pwm 67 vs 4', np.mean(diff_pwm_64_4[per_6]))

print('mean relative period 1 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_1]))
print('mean relative period 2 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_2]))
print('mean relative period 3 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_3]))
print('mean relative period 4 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_4]))
print('mean relative period 5 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_5]))
print('mean relative period 6 pwm 67 vs 17', np.mean(diff_pwm_64_16[per_6]))

plt.yscale('log')
plt.xlabel('Time [s]', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower left', prop={'size': fonts})
plt.savefig("joule_losses_space_relative.jpg", bbox_inches='tight')
plt.show()

# plt.plot(t_4k_pwm[1:], diff_sin_64_4, label='Relativ error sin 67 vs 4')
# plt.plot(t_4k_pwm[1:], diff_sin_64_16, label='Relativ error sin 67 vs 17')
# print('mean relativ sin 67 vs 4', np.mean(diff_sin_64_4))
# print('mean relativ sin 67 vs 17', np.mean(diff_sin_64_16))
#
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()
