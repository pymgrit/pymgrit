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

# filelist = glob.glob(os.path.join('/home/jens/uni/results/induction_machine_jl_4k_nonlinear_1per_214', '*'))
# sol_1024, t_1024, runtime_1024 = solution_new(filelist)
# #fac = 0.5* (1-np.cos(np.pi*t_1024/(2*0.02)))
# sol_1024 = np.array(sol_1024)
# plt.plot(t_1024, sol_1024, label = '1024')
# plt.show()

filelist = glob.glob(os.path.join('/home/jens/uni/pasirom/python_mgrit/cluster_runs/results/ind_mac_nonlinear', '*'))
sol2, t2, r2 = solution_new(filelist)
#fac = 0.5* (1-np.cos(np.pi*t_1024/(2*0.02)))
sol2 = np.array(sol2)
plt.plot(t2, sol2, label = '1024')
plt.show()