import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from operator import itemgetter


def solution_new(filelist):
    sol = []
    stop = []
    jl = []
    ia = []
    ib = []
    ic = []
    ua = []
    ub = []
    uc = []
    back3 = []
    back4 = []
    back5 = []
    back12 = []
    back13 = []
    back14 = []
    for infile in sorted(filelist):
        a = np.load(infile, allow_pickle=True).item()
        sol.append([float(re.split(r'(?<!e)-', infile)[-1][:-4]),
                    a['jl'],
                    a['ia'],
                    a['ib'],
                    a['ic'],
                    a['ua'],
                    a['ub'],
                    a['uc'],
                    a['back3'],
                    a['back4'],
                    a['back5'],
                    a['back12'],
                    a['back13'],
                    a['back14'],
                    ])
        t = a['t']
        runtime = a['time']
    sol.sort(key=lambda x: x[0])
    for i in range(len(sol)):
        jl.append(sol[i][1])
        ua.append(sol[i][2])
        ub.append(sol[i][3])
        uc.append(sol[i][4])
        ia.append(sol[i][5])
        ib.append(sol[i][6])
        ic.append(sol[i][7])
        back3.append(sol[i][8])
        back4.append(sol[i][9])
        back5.append(sol[i][10])
        back12.append(sol[i][11])
        back13.append(sol[i][12])
        back14.append(sol[i][13])
    jl = [item for sublist in jl for item in sublist]
    ia = [item for sublist in ia for item in sublist]
    ib = [item for sublist in ib for item in sublist]
    ic = [item for sublist in ic for item in sublist]
    ua = [item for sublist in ua for item in sublist]
    ub = [item for sublist in ub for item in sublist]
    uc = [item for sublist in uc for item in sublist]
    back3 = [item for sublist in back3 for item in sublist]
    back4 = [item for sublist in back4 for item in sublist]
    back5 = [item for sublist in back5 for item in sublist]
    back12 = [item for sublist in back12 for item in sublist]
    back13 = [item for sublist in back13 for item in sublist]
    back14 = [item for sublist in back14 for item in sublist]
    return {'t': t, 'runtime': runtime, 'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc,
            'back3': back3, 'back4': back4, 'back5': back5, 'back12': back12, 'back13': back13, 'back14': back14, }
    return sol, t, runtime


filelist = glob.glob(os.path.join('/home/jens/uni/results/induction_machine_jl_sin', '*'))
res_dic = solution_new(filelist)
# fac = 0.5* (1-np.cos(np.pi*t_1024/(2*0.02)))
plt.plot(res_dic['t'], np.array(res_dic['jl']), label='1024')
plt.show()

plt.plot(res_dic['t'], np.array(res_dic['back3']), label='1024')
plt.plot(res_dic['t'], np.array(res_dic['back12']), label='1024')
plt.plot(res_dic['t'], np.array(res_dic['ia']), label='1024')
plt.plot(res_dic['t'], np.array(res_dic['ua']), label='1024')
plt.show()