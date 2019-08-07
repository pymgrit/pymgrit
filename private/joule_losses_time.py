import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from operator import itemgetter


def solution_new(filelist):
    sol = []
    stop = []
    for infile in sorted(filelist):
        a = np.load(infile, allow_pickle=True).item()
        sol.append([float(re.split(r'(?<!e)-', infile)[-1][:-4]), a['jl']])
        t = a['t']
        runtime = a['time']
    sol.sort(key=lambda x: x[0])
    for i in range(len(sol)):
        sol[i] = sol[i][1]
    sol = [item for sublist in sol for item in sublist]
    return sol, t, runtime


filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-1024-pwm-4k', '*'))
sol_1024, t_1024, runtime_1024 = solution_new(filelist)
sol_1024 = np.array(sol_1024)
# plt.plot(t_1024, sol_1024, label = '1024')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-2048-pwm-4k', '*'))
sol_2048, t_2048, runtime_2048 = solution_new(filelist)
sol_2048 = np.array(sol_2048)
# plt.plot(t_2048[::2], sol_2048[::2], label = '2048')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-4096-pwm-4k', '*'))
sol_4096, t_4096, runtime_4096 = solution_new(filelist)
sol_4096 = np.array(sol_4096)
# plt.plot(t_4096[::4], sol_4096[::4], label = '4096')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-8192-pwm-4k', '*'))
sol_8192, t_8192, runtime_8192 = solution_new(filelist)
sol_8192 = np.array(sol_8192)
# plt.plot(t_8192[::8], sol_8192[::8], label = '8192')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-16384-pwm-4k', '*'))
sol_16384, t_16384, runtime_16384 = solution_new(filelist)
sol_16384 = np.array(sol_16384)
# plt.plot(t_16384[::16], sol_16384[::16], label = '16384')

diff_sol_16384_sol_8192 = np.abs(sol_16384[::2] - sol_8192)
diff_sol_16384_sol_4096 = np.abs(sol_16384[::4] - sol_4096)
diff_sol_16384_sol_2048 = np.abs(sol_16384[::8] - sol_2048)
diff_sol_16384_sol_1024 = np.abs(sol_16384[::16] - sol_1024)


print('runtimes', runtime_1024, runtime_2048,runtime_4096,runtime_8192,runtime_16384)

fonts=18

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
plt.plot(t_8192, diff_sol_16384_sol_8192, label='Absolute error 16384 vs 8192 on identical points', lw=2)
plt.plot(t_4096, diff_sol_16384_sol_4096, label='Absolute error 16384 vs 4096 on identical points', lw=2)
plt.plot(t_2048, diff_sol_16384_sol_2048, label='Absolute error 16384 vs 2048 on identical points', lw=2)
plt.plot(t_1024, diff_sol_16384_sol_1024, label='Absolute error 16384 vs 1024 on identical points', lw=2)
# print('mean absolute 16384 vs 8192', np.mean(diff_sol_16384_sol_8192))
# print('mean absolute 16384 vs 4096', np.mean(diff_sol_16384_sol_4096))
# print('mean absolute 16384 vs 2048', np.mean(diff_sol_16384_sol_2048))
# print('mean absolute 16384 vs 1024', np.mean(diff_sol_16384_sol_1024))
print('max absolute 16384 vs 8192', np.max(diff_sol_16384_sol_8192))
print('max absolute 16384 vs 4096', np.max(diff_sol_16384_sol_4096))
print('max absolute 16384 vs 2048', np.max(diff_sol_16384_sol_2048))
print('max absolute 16384 vs 1024', np.max(diff_sol_16384_sol_1024))
plt.yscale('log')
plt.xlabel('Time / s', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower right', prop={'size': fonts})
plt.savefig("joule_losses_time_absolute.jpg", bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
diff_sol_16384_sol_8192 = np.abs(sol_16384[::2][1:] - sol_8192[1:]) / sol_16384[::2][1:]
diff_sol_16384_sol_4096 = np.abs(sol_16384[::4][1:] - sol_4096[1:]) / sol_16384[::4][1:]
diff_sol_16384_sol_2048 = np.abs(sol_16384[::8][1:] - sol_2048[1:]) / sol_16384[::8][1:]
diff_sol_16384_sol_1024 = np.abs(sol_16384[::16][1:] - sol_1024[1:]) / sol_16384[::16][1:]
plt.plot(t_8192[1:], diff_sol_16384_sol_8192, label='Relativ error 16384 vs 8192 on identical points', lw=2)
plt.plot(t_4096[1:], diff_sol_16384_sol_4096, label='Relativ error 16384 vs 4096 on identical points', lw=2)
plt.plot(t_2048[1:], diff_sol_16384_sol_2048, label='Relativ error 16384 vs 2048 on identical points', lw=2)
plt.plot(t_1024[1:], diff_sol_16384_sol_1024, label='Relativ error 16384 vs 1024 on identical points', lw=2)
# print('mean relativ 16384 vs 8192', np.mean(diff_sol_16384_sol_8192))
# print('mean relativ 16384 vs 4096', np.mean(diff_sol_16384_sol_4096))
# print('mean relativ 16384 vs 2048', np.mean(diff_sol_16384_sol_2048))
# print('mean relativ 16384 vs 1024', np.mean(diff_sol_16384_sol_1024))
print('max relativ 16384 vs 8192', np.max(diff_sol_16384_sol_8192))
print('max relativ 16384 vs 4096', np.max(diff_sol_16384_sol_4096))
print('max relativ 16384 vs 2048', np.max(diff_sol_16384_sol_2048))
print('max relativ 16384 vs 1024', np.max(diff_sol_16384_sol_1024))
plt.yscale('log')
plt.xlabel('Time / s', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower right', prop={'size': fonts})
plt.savefig("joule_losses_time_relative.jpg", bbox_inches='tight')
plt.show()
