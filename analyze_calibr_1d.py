# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:03:42 2020

@author: 03125327
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
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
    n_elements_to_pick = np.array(ci_percents)/100.*N_POSTERIOR_SAMPLES 
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

#%%
"""
 Corner plot
"""
def corner_plot(samples, savefig=True):
    import corner
    labels = ['s0', 's1', 't0', 't1', 't2']
    fig = corner.corner(samples, labels=labels)
    if savefig:
        fig.savefig("MCMC_corner_result.png")
    
    return 0
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
            