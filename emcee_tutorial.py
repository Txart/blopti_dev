# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:55:05 2020

@author: 03125327
"""


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

plt.close('all')

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294


# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

# Linear least squares
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

# loglikelihood 
def log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# optimized likelihood
from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))


plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
# plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

def log_prior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


import emcee

N_WALKERS = 5
# Initialize walkers in maximum likelihood result
pos = soln.x + 1e-4 * np.random.randn(N_WALKERS, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True);


#%%
# Take a look at walkers
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

tau = sampler.get_autocorr_time()
print(tau)

#%%
# Discard 100 samples
flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)


#%%
# Corner plot 
import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=[m_true, b_true]
);

#%%
# and plot over data

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

#%%
# Multiprocessing
multi= False
if multi==True:
    import numpy as np
    import time
    import emcee
    from multiprocessing import Pool
    
    np.random.seed(42)
    initial = np.random.randn(32, 5)
    nwalkers, ndim = initial.shape
    nsteps = 100
    
    data = np.random.randn(5000, 200)
    
    def log_prob_data(theta, data):
        a = data[0]  # Use the data somehow...
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5 * np.sum(theta ** 2)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_data, args=(data,))
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    serial_data_time = end - start
    print("Serial took {0:.1f} seconds".format(serial_data_time))
    
    
    def log_prob_data_global(theta):
        a = data[0]  # Use the data somehow...
        t = time.time() + np.random.uniform(0.005, 0.008)
        while True:
            if time.time() >= t:
                break
        return -0.5 * np.sum(theta ** 2)
    
    
    
    with Pool(4) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_data, pool=pool, args=(data,)
        )
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_data_global_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_data_global_time))
        print(
            "{0:.1f} times faster than serial".format(
                serial_data_time / multi_data_global_time
            )
        )
