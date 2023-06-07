import os
import sys
import site 

site.addsitedir('../covid/py')

import numpy as np
import pandas as pd
from covid_configs import *
import globals

SCOPE = "DE"
if len(sys.argv) >= 2:
  SCOPE = sys.argv[1]
globals.initialize()
setpop(SCOPE)
print (globals.npop)

#filecode = sys.argv[-1]

chainsfile = "chains_" + SCOPE + ".npy"
flat_samples = np.load(chainsfile)
print(chainsfile)

ndim = flat_samples.shape[1]
labels = makelabels(ndim)

parameter = []
median = []
perc25 = []
perc75 = []
print("parameter","median","25th","75th")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
#    txt = "{3},{0:.3f},{1:.3f},{2:.3f}"
#    txt = txt.format(mcmc[1], mcmc[1]-q[0], mcmc[1]+q[1], labels[i])
#    print(txt)
    parameter.append(labels[i])
    median.append(np.round(mcmc[1],3))
    perc25.append(np.round(mcmc[1]-q[0],3))
    perc75.append(np.round(mcmc[1]+q[1],3))

print("")
listoftuples = list(zip(parameter,median,perc25,perc75))
df = pd.DataFrame(listoftuples, columns=["parameter","median","25th","75th"])
print(df)

outfilename = str("results_" + SCOPE + ".csv")
#outfilepath = os.path.join('./output/', outfilename)

df.to_csv(outfilename, index=False)
