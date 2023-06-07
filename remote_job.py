import os
import sys
import time
from time import strftime
import numpy as np
import pandas as pd
import emcee
import corner
from scipy.optimize import minimize
import random
import csv

import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def file_cleanup(path):
    now = time.time()

    i = 0
    for f in os.listdir(path):
        ff = os.path.join(path, f)
        if os.stat(ff).st_mtime < now - 3 * 86400:
            os.remove(ff)
            i += 1
    return i

# start with file cleanup
file_cleanup('./input')
file_cleanup('./output')

filename = sys.argv[1]
nsteps = sys.argv[2]
email = sys.argv[3]

filepath = str('./input/' + filename)
df = pd.read_csv(filepath)

x = np.array(df['x'])
y = np.array(df['y'])
yerr = np.array(df['yerr'])

# linear least squares solution
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))

# Maximum likelihood solution
nll = lambda *args: -log_likelihood(*args)
initial = np.array([w[0], w[1], np.log(0.534)]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml, log_f_ml = soln.x

# MCMC
pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, nsteps, progress=True)

try:
    tau = sampler.get_autocorr_time()
except:
    tau = [29.5, 29.1, 28.6]

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

mcmc_m = np.percentile(flat_samples[:, 0], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])
mcmc_b = np.percentile(flat_samples[:, 1], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])
mcmc_f = np.percentile(flat_samples[:, 2], [50, 16, 84, 2.5, 97.5, 0.15, 99.85])

timestamp = str(strftime("%Y%m%d_%H%M%S"))
randstamp = str(random.randint(0, 100000))
outfilename = str("results_" + timestamp + "_" + randstamp + ".csv")
outfilepath = os.path.join('./output/', outfilename)

lsarr = [w[0], np.sqrt(cov[0, 0]), w[1], np.sqrt(cov[1, 1])]
header = '\n'.join(
     [line for line in
          ['Least Squares',
           'm,m_err,b,b_err',
             ','.join(map(str, lsarr)),
             'Maximum Likelihood',
             'm,b,log(f)',
             ','.join(map(str, soln.x)),
             'MCMC,' + str(nsteps),
             'var,median,1sigma_m,1sigma_p,2sigma_m,2sigma_p,3sigma_m,3sigma_p',
             'm,' + ','.join(map(str, mcmc_m)),
             'b,' + ','.join(map(str, mcmc_b)),
             'f,' + ','.join(map(str, mcmc_f)),
             'Raw Data',''
            ]
        ]
    )

with open(outfilepath, 'w', newline='') as csvfile:
    for line in header:
        csvfile.write(line)
    df.to_csv(csvfile, index=False)

print(outfilename)


