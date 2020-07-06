# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:37:16 2020

@author: 03125327
"""
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

#%%
# read mcmc data from hdf5 file
fname = "output/mcmc_result_chain.h5"
reader = emcee.backends.HDFBackend(fname)

samples = reader.get_chain()
ndim = samples.shape[-1]
labels = ["t0", 't1', 't2',  "s0", 's1', 's2']

#%%
# Take a look at walkers
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

if len(labels)!=ndim:
    raise ValueError('Different number of labels and of ndim')

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

# Autocorrelation time
tau = reader.get_autocorr_time()
print('tau: ', tau)

#%%
# Discard  samples
flat_samples = reader.get_chain(discard=200, thin=100, flat=True)


#%%
# Corner plot 

fig = corner.corner(
    flat_samples, labels=labels
);

#%%
# and plot over data

# inds = np.random.randint(len(flat_samples), size=100)
# for ind in inds:
#     sample = flat_samples[ind]
#     plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
# plt.legend(fontsize=14)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y");