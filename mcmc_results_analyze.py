# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:44:11 2020

@author: 03125327
"""
import emcee
import corner
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Math

reader = emcee.backends.HDFBackend("mcmc_result_chain.h5", read_only=True)
flatchain = reader.get_chain(flat=True)
fatchain = reader.get_chain(flat=False)



true_params = [0.1, 0.2, 1., 0.01, 1.1]

#%%
# Take a look at walkers
fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = reader.get_chain()
labels = ["s0", "s1", "t0", "t1", "t2"]
for i in range(5):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.hlines(y=true_params[i], xmin=0, xmax=samples.shape[0], colors='blue')
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

tau = reader.get_autocorr_time()
print(tau)

#%%
# Acceptance fraction
chainlength = fatchain.shape[0]
walker_acceptance_fraction = reader.accepted/chainlength
print(f'Mean of accepted steps = {np.mean(walker_acceptance_fraction)}')

# N of infinities
logprob = reader.get_log_prob()
print(f'Fraction of infinity logprobs = {np.count_nonzero(logprob==-np.inf)/logprob.size}')


#%%
# corner_plot

flatchain = reader.get_chain(flat=True, discard=1000)
labels = ['s0', 's1', 't0', 't1', 't2']
fig = corner.corner(
    flatchain, labels=labels, truths=true_params
);

#%%
# Percentile threshold


for i in range(5):
    mcmc = np.percentile(flatchain[:, i], [16, 68, 95])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))