import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from operator import itemgetter


def plot(file):
    load = np.load(file, allow_pickle=True).item()
    u = load['u']

    jl = np.zeros(len(u))

    for i in range(len(u)):
        jl[i] = u[i].jl

    plt.plot(load['t'], jl)


def plot_solution(t, sol):
    jl = np.zeros(len(sol))

    for i in range(len(sol)):
        jl[i] = sol[i].jl

    plt.plot(t, jl)


def solution(filelist):
    sol = []
    for infile in sorted(filelist):
        a = np.load(infile, allow_pickle=True).item()
        sol.append([a['u']])
        t = a['t']
    sol = [item for sublist in sol for item in sublist]
    return sol, t

def solution_new(filelist):
    sol = []
    stop = []
    for infile in sorted(filelist):
        a = np.load(infile, allow_pickle=True).item()
        sol.append([float(re.split(r'(?<!e)-', infile)[-1][:-4]), a['jl']])
        t = a['t']
    sol.sort(key=lambda x: x[0])
    for i in range(len(sol)):
        sol[i] = sol[i][1]
    sol = [item for sublist in sol for item in sublist]
    return sol, t


# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-4k', '*'))
# sol, t = solution(filelist)
# jl_solution(t, sol)
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-4k', '*'))
# sol, t = solution(filelist)
# jl_solution(t, sol)
#
filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-sin-16k', '*'))
sol, t = solution(filelist)
plot_solution(t, sol)

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.125-16385-pwm-16k', '*'))
sol, t = solution(filelist)
plot_solution(t, sol)

# filelist = glob.glob(os.path.join('/home/jens/uni/results/2019-07-29|23:01:31', '*'))
# sol, t = solution(filelist)
# jl_solution(t, sol)

# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-128-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '128')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-256-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t[::2], np.array(sol)[::2], label = '256')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-512-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t[::4], np.array(sol)[::4], label = '512')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-1024-pwm-4k', '*'))
sol, t = solution_new(filelist)
plt.plot(t, np.array(sol), label = '1024')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-2048-pwm-4k', '*'))
sol, t = solution_new(filelist)
plt.plot(t[::2], np.array(sol)[::2], label = '2048')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-4096-pwm-4k', '*'))
sol, t = solution_new(filelist)
plt.plot(t[::4], np.array(sol)[::4], label = '4096')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-8192-pwm-4k', '*'))
sol, t = solution_new(filelist)
plt.plot(t[::8], np.array(sol)[::8], label = '8192')

filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-16384-pwm-4k', '*'))
sol, t = solution_new(filelist)
plt.plot(t[::16], np.array(sol)[::16], label = '16384')

# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-128-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '128')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-256-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '256')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-512-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '512')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-1024-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '1024')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-2048-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '2048')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-4096-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '4096')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-8192-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '8192')
#
# filelist = glob.glob(os.path.join('/home/jens/uni/results/t0-0.0039-16384-pwm-4k', '*'))
# sol, t = solution_new(filelist)
# plt.plot(t, np.array(sol), label = '16384')

plt.legend(loc='upper left')
plt.show()
