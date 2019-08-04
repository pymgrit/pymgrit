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


filelist = glob.glob(os.path.join('/home/jens/uni/results/induction_machine_jl_4k', '*'))
sol_4, t_4, runtime_4 = solution_new(filelist)
sol_4 = np.array(sol_4)

filelist = glob.glob(os.path.join('/home/jens/uni/results/induction_machine_jl_17k', '*'))
sol_17, t_17, runtime_17 = solution_new(filelist)
sol_17 = np.array(sol_17)

filelist = glob.glob(os.path.join('/home/jens/uni/results/induction_machine_jl_69k', '*'))
sol_69, t_69, runtime_69 = solution_new(filelist)
sol_69 = np.array(sol_69)

diff_sol_69_sol_17 = np.abs(sol_69 - sol_17)
diff_sol_69_sol_4 = np.abs(sol_69 - sol_4)

print('runtimes', runtime_4, runtime_17, runtime_69)

fonts = 18

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
plt.plot(t_4, diff_sol_69_sol_17, label='Absolute error sin 69 vs 17', lw=2)
plt.plot(t_4, diff_sol_69_sol_4, label='Absolute error sin 69 vs 4', lw=2)
print('mean absolute 69 vs 17', np.mean(diff_sol_69_sol_17))
print('mean absolute 69 vs 4', np.mean(diff_sol_69_sol_4))

print('max absolute 69 vs 17', np.max(diff_sol_69_sol_17))
print('max absolute 69 vs 4', np.max(diff_sol_69_sol_4))

plt.yscale('log')
plt.xlabel('Time / s', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower right', prop={'size': fonts})
plt.savefig("joule_losses_space_absolute.jpg", bbox_inches='tight')
plt.show()

diff_sol_69_sol_17 = np.abs(sol_69 - sol_17)[1:] / sol_69[1:]
diff_sol_69_sol_4 = np.abs(sol_69 - sol_4)[1:] / sol_69[1:]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)
plt.plot(t_4, diff_sol_69_sol_17, label='Relative error sin 69 vs 17', lw=2)
plt.plot(t_4, diff_sol_69_sol_4, label='Relative error sin 69 vs 4', lw=2)
print('mean relative 69 vs 17', np.mean(diff_sol_69_sol_17))
print('mean relative 69 vs 4', np.mean(diff_sol_69_sol_4))

print('max relative 69 vs 17', np.max(diff_sol_69_sol_17))
print('max relative 69 vs 4', np.max(diff_sol_69_sol_4))

plt.yscale('log')
plt.xlabel('Time / s', fontsize=fonts, weight='bold')
plt.ylabel('Difference', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='lower right', prop={'size': fonts})
plt.savefig("joule_losses_space_relative.jpg", bbox_inches='tight')
plt.show()
