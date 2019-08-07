import glob
import os
import numpy as np

filelist = glob.glob(os.path.join('/home/jens/uni/pasirom/python_mgrit/results/2019-07-28|11:57:08', '*'))
sol = []
for infile in sorted(filelist):
    a = np.load(infile, allow_pickle=True).item()
    sol.append(a['u'])
sol = [item for sublist in sol for item in sublist]