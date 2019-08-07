import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from operator import itemgetter

def jl_solution(sol):
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
    sol = [item for sublist in sol for item in sublist]
    return sol, t

fonts = 18

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
plt.grid(True)

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-4k', '*'))
sol, t = solution(filelist)
jl = jl_solution(sol)
plt.plot(t, jl, label='PWM signal', lw=2)

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-4k', '*'))
sol, t = solution(filelist)
jl = jl_solution(sol)
plt.plot(t, jl, label='Sinus signal', lw=2, linestyle='--')


plt.xlabel('Time [s]', fontsize=fonts, weight='bold')
plt.ylabel('Joule losses [?]', fontsize=fonts, weight='bold')
plt.xticks(size=fonts, weight='bold')
plt.yticks(size=fonts, weight='bold')
plt.legend(loc='upper right', prop={'size': fonts}, handlelength = 4)
plt.savefig("joule_losses_over_periods.jpg", bbox_inches='tight')
plt.show()