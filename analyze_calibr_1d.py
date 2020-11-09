# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:03:42 2020

@author: 03125327
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

#%%
def sample_from_posterior(mcmc_flat_chain, n_samples, mcmc_logprob):
    """
    Takes a number of parameter samples from the posterior distribution, and computes
    the hydrology with them.
    Useful for checking model performance on the testing set (i.e., on runs over
    data not used for parameter calibration). For instance, for approximating
    statistics of confidence intervals and the like.
        - 
    Parameters
    ----------
    mcmc_flat_chain : numpy array
        MCMC chain, flattened. What you get when running emcee.get_chain(flat=True)
    n_samples : int
        Number of samples to take from the posterior and, thus, number of times
        to run the hydro
    mcmc_logprob : numpy array
        MCMC log probabilities. What you get when running emcee.get_log_prob(flat=True)

    Raises
    ------
    ValueError
        When more samples want to be drawn than there exist in the posterior

    Returns
    -------
    list
        Samples from the posterior without infinities

    """

    if n_samples > len(mcmc_flat_chain):
        raise ValueError("There are not enough samples in the MCMC to draw from.")
    
    # choose random samples from posterior that dont have posterior=-inf
    randints = []
    while len(randints)<100:    
        rint = np.random.randint(low=0, high=len(mcmc_flat_chain))
        if np.isfinite(mcmc_logprob[rint]):
            randints.append(rint)
        else:
            continue
        
    post_params = [mcmc_flat_chain[ind] for ind in randints]
    
    return post_params

def confidence_intervals(ci_percents, n_sensors, ndays, n_posterior_samples):
    """
    Compute CI from posterior-sampled-hydro.
    TODO: Not a definitive setup with the dimensions of the arrays, couldn't think
    of  one that was generalizable, has to be tailored for each case.

    Parameters
    ----------
    ci_percents : list
        List of confidence intervals to compute
    n_sensors : int
        Number of sensors
    ndays : int
        ndays the hydro model has run

    Returns
    -------
    conf_interv_results : numpy array
        Call as follows: conf_interv_results[which sensor?][which day?][which CI?]
        From there: -> [0] is maximum of the CI
                       [1] is minimum of the CI

    """

    # translate CI percentages to how many items to pick up from array
    n_elements_to_pick = np.array(ci_percents)/100.*n_posterior_samples 
    n_elements_to_pick = n_elements_to_pick.astype(int)
    
    conf_interv_results = np.zeros(shape=(n_sensors, ndays, len(ci_percents), 2))
    for i in range(n_sensors):
        modelled = results[:,i]
        diff = np.abs(modelled - measurements.T[i])
        for day in range(NDAYS):
            model_day = modelled[:, day]
            diff_day = diff[:, day]
            model_diffsorted = model_day[diff_day.argsort()] # modelled values sorted by absolute difference
            for n, nele in enumerate(n_elements_to_pick):
                conf_interv_results[i,day,n][0] = np.max(model_diffsorted[:nele])
                conf_interv_results[i,day,n][1] = np.min(model_diffsorted[:nele])
    
    return conf_interv_results
    

#%%   
"""
 read from backend
"""
def read_from_backend(filename):
    
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    samples = reader.get_chain(discard=0, thin=1, flat=True)
    print(f"Read from backend. \n   Samples shape = {samples.shape}")
    logprobs = reader.get_log_prob(flat=True)
    frac_infinities = 1 - np.count_nonzero(np.isfinite(logprobs))/samples.shape[0]
    print(f"   Fraction of infinities = {frac_infinities * 100:.2f}%")
    
    return reader

fname = r"mcmc_result_chain.h5"
# fname = "mcmc_result_chain.h5"
reader = read_from_backend(fname)
fat_samples = reader.get_chain(flat=False)

#%% 
# Take a look at walkers and autocorrelation time

nparams = fat_samples.shape[2]
fig, axes = plt.subplots(nparams, sharex=True)

labels = ["s1", "s2", 't1', 't2']
for i in range(nparams):
    ax = axes[i]
    ax.plot(fat_samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(fat_samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

tau = reader.get_autocorr_time()
print(tau)

#%%
# Thin and discard samples
DISCARD = 0
THIN = 1
flat_samples = reader.get_chain(discard=DISCARD, thin=THIN, flat=True)
print(f"flat samples shape: {flat_samples.shape}")

#%%
"""
 Corner plot
"""
def corner_plot(samples, savefig=True):
    import corner
    labels = ['s1', 's2', 't1', 't2']
    fig = corner.corner(samples, labels=labels)
    if savefig:
        fig.savefig("MCMC_corner_result.png")
    
    return 0

corner_plot(flat_samples, savefig=True)

#%%
# ML and ranges for values
flat_samples_all = reader.get_chain(flat=True)
log_probs_all = reader.get_log_prob(flat=True)
max_lp = np.max(log_probs_all)
max_likelihood_params = flat_samples_all[np.where(log_probs_all==max_lp)[0]][0]
   

#%%
"""
 Plot Sy and T
"""
def Sy(zeta, s1, s2):
    return np.exp(s1 + s2 * zeta)
    
def T(zeta, t1, t2, b):
    return np.exp(t1)/t2 * (np.exp(t2*zeta) - np.exp(t2*b))

n_samples = len(flat_samples)

# Conf intervals
conf_int_perc = np.array([90,10]) # % of ci to be computed. Put largest first
conf_int_indices_to_pick = (100. - conf_int_perc) * n_samples / 100
conf_int_indices_to_pick = conf_int_indices_to_pick.astype(dtype=int)
log_probs = reader.get_log_prob(flat=True, discard=DISCARD, thin=THIN)
sorted_flatsamples = flat_samples[-log_probs.argsort()] # samples sorted by log_probs


zeta_grid = np.arange(-3, 1, 0.01)
# h_2d_grid = np.array([h_grid,]*n_samples)
sto_array = np.array([Sy(zeta_grid, s1, s2) for s1, s2, _, _ in sorted_flatsamples])
# TODO: pass also b.
tra_array = np.array([T(zeta_grid, t1, t2) for _, _, t1, t2  in sorted_flatsamples])

sto_ci = [0] * len(conf_int_perc) # [[Smin_ci1, Smax_ci1], [Smin_ci2, Smax_ci2], ...]
tra_ci = [0] * len(conf_int_perc)
for i, ci in enumerate(conf_int_indices_to_pick):
    # storage
    sto_arr_ci = sto_array[:ci]
    sto_max_ci = sto_arr_ci.max(axis=0)
    sto_min_ci = sto_arr_ci.min(axis=0)
    sto_ci[i] = [sto_min_ci, sto_max_ci]
    # transmissivity
    tra_arr_ci = tra_array[:ci]
    tra_max_ci = tra_arr_ci.max(axis=0)
    tra_min_ci = tra_arr_ci.min(axis=0)
    tra_ci[i] = [tra_min_ci, tra_max_ci]
       
# Plot all curves with some alpha    
fig, ax = plt.subplots(nrows=2, ncols=1)
axS = ax[0]; axT = ax[1]
axS.set_title('Sy(zeta)'); axT.set_title('T(zeta)')
axS.set_xlim(left=0, right=1.5); axT.set_xlim(left=0, right=10)
axS.set_xlabel('Sy'); axS.set_ylabel('zeta(m)')
axT.set_xlabel('T'); axT.set_ylabel('zeta(m)')

for s, t in zip(sto_array, tra_array):
    # axS.plot(s, h_grid, alpha=1/(30*np.log(flat_samples.shape[0])), color='black')
    # axT.plot(t, h_grid, alpha=1/(30*np.log(flat_samples.shape[0])), color='black')
    axS.plot(s, zeta_grid, alpha=1/256, color='black')
    axT.plot(t, zeta_grid, alpha=1/256, color='black')

axS.hlines(y=0, xmin=0, xmax=1, colors='brown', linestyles='dashed', label='peat surface')
axT.hlines(y=0, xmin=0, xmax=10, colors='brown', linestyles='dashed', label='peat surface')

# Plot confidence intervals and ML curves for S and T
fig, ax = plt.subplots(nrows=2, ncols=1)
# CI
axS = ax[0]; axT = ax[1]
axS.set_xlim(left=0, right=1.5); axT.set_xlim(left=0, right=10)
axS.set_xlabel('Sy'); axS.set_ylabel('zeta(m)')
axT.set_xlabel('T'); axT.set_ylabel('zeta(m)')

for i, ci in enumerate(conf_int_perc):
    axS.fill_betweenx(zeta_grid, sto_ci[i][1], sto_ci[i][0], label= f'{ci}% conf. int.', alpha=0.1)
    axT.fill_betweenx(zeta_grid, tra_ci[i][1], tra_ci[i][0], label= f'{ci}% conf. int.', alpha=0.1)

# ML
s1ML = max_likelihood_params[0]
s2ML = max_likelihood_params[1]
t1ML = max_likelihood_params[2]
t2ML = max_likelihood_params[3]

axS.plot(Sy(zeta_grid, s1ML, s2ML), zeta_grid, label='ML')
axT.plot(T(zeta_grid,  t1ML, t2ML), zeta_grid, label='ML')

fig.legend()

axS.hlines(y=0, xmin=0, xmax=1, colors='brown', linestyles='dashed', label='peat surface')
axT.hlines(y=0, xmin=0, xmax=10, colors='brown', linestyles='dashed', label='peat surface')


#%%     
"""
 Sample from resulting posterior.
 TODO: CHANGE THESE FOR NEW HYDRO ETC.
"""
def xxxxxxxxxxxx(samples, logprobs):
    N_POSTERIOR_SAMPLES = 10
    post_params = sample_from_posterior(samples, N_POSTERIOR_SAMPLES, logprobs)
                          
    #%%
    """
    Run hydro with set of params from the posterior
    """
    import calibration_1d
    
    bcleft, bcright, measurements, days, precip, evapotra = calibration_1d.read_sensors('fabricated_data.txt')
    boundary = np.array(list(zip(bcleft, bcright)))
    HINI = 8.; nx = 100; dx = 1.; dt = 1.; NDAYS = len(measurements);
    SENSOR_LOCATIONS = [0, 12, 67, 99];
    
    results = []
    for params in post_params:
        s0 = params[0]; s1 = params[1] 
        t0 = params[2]; t1 = params[3]; t2 = params[4];
        
        theta_ini = np.exp(s0 + s1*HINI) 
        try:
            simulated_wtd = calibration_1d.hydro_1d(nx, dx, dt, params, theta_ini,
                                                    NDAYS, SENSOR_LOCATIONS, boundary, precip, evapotra)
        except:
            print("PROBLEM IN HYDRO WITH PARAMS")
            continue
        
        results.append(simulated_wtd.T) # NOTE, appending the transpose!!
      
    results = np.array(results)
    #%%
    """
     Compute confidence intervals
    """
    ci_percents = [20, 50, 90]
    n_sensors = len(SENSOR_LOCATIONS)
    conf_interv_results = confidence_intervals(ci_percents, n_sensors, NDAYS, N_POSTERIOR_SAMPLES)
    
    
    #%%
    """
     Plot over data, sensor by sensor:
       - samples from posterior WTD over data
       - Confidence intervals
    """
                
    fig,axes = plt.subplots(nrows=2, ncols=2)
    axes = axes.ravel()
    
    fig_ci, axes_ci = plt.subplots(nrows=2, ncols=2)
    axes_ci = axes_ci.ravel()
    
    for sensor_n, sensor_loc in enumerate(SENSOR_LOCATIONS):
        sensor_simulated_wtd = results[:,sensor_n] # order results by sensors 
        measured_value_in_sensor = measurements.T[sensor_n]
        days_x_axis = list(range(0,len(measured_value_in_sensor)))
        for simu_sensor in sensor_simulated_wtd:
            axes[sensor_n].plot(days_x_axis, simu_sensor, 'orange', alpha=0.1)
        
        # confidence intervals
        colors = ['blue', 'red', 'yellow']
        if len(colors) < len(ci_percents):
            raise ValueError("Too few colors")
        for nci, ci in enumerate(ci_percents):
            top_ci = conf_interv_results[sensor_n][:,nci][:,0]
            bottom_ci = conf_interv_results[sensor_n][:,nci][:,1]
            axes_ci[sensor_n].fill_between(days_x_axis, top_ci, bottom_ci,
                                           label=f"{ci}% CI", alpha=0.2, color=colors[nci])
            axes_ci[sensor_n].plot(days_x_axis, top_ci, color=colors[nci], linewidth=0.2)
            axes_ci[sensor_n].plot(days_x_axis, bottom_ci, color=colors[nci], linewidth=0.2)
        
            
        
        # true, measured values    
        axes[sensor_n].plot(days_x_axis, measured_value_in_sensor, 'black')
        axes_ci[sensor_n].plot(days_x_axis, measured_value_in_sensor, 'black')
        
        axes[sensor_n].set_title(f"sensor_n {sensor_n} at location {sensor_loc}")
    
    fig_ci.legend()
    plt.show()
            