# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:30:26 2020

@author: 03125327

T(h) and Sy(h) calibration comparing measurements of WTD along transects
with modelled 1-D Boussinesq equation.

NOTES:
    - Peat depth assumed equal along the transect

"""
import argparse
import fipy as fp
from fipy.tools import numerix
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee

import get_data
import hydro_calibration


#%%
"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run MCMC parameter estimation')
 
parser.add_argument('--ncpu', default=1, help='(int) Number of processors', type=int)
parser.add_argument('-cl','--chainlength', default=1, help='(int) Length of MCMC chain', type=int)
parser.add_argument('-w','--nwalkers', default=8, help='(int) Number of walkers in parameter space', type=int)
args = parser.parse_args()
 
N_CPU = args.ncpu
MCMC_STEPS = args.chainlength
N_WALKERS = args.nwalkers


#%%
"""
 Get data from measurements
 Choose relevant transects.

""" 
fn_weather_data = Path('data/weather_station_historic_data.xlsx')
fn_wtd_data = Path('data/historic_wtd_data_18-1-2021.xlsx')
fn_transect_info = Path('data/transect_info.xlsx')
fn_selected_dates = Path('data/transect_selected_dates.xlsx')
transect_info = pd.read_excel(fn_transect_info)
transect_selected_dates = pd.read_excel(fn_selected_dates)
dfs_by_transects = get_data.main(fn_weather_data, fn_wtd_data, api_call=False)


#%%

"""
MCMC parameter estimation
""" 
# Parameters
dx = 5 # in m
dt = 1. # in days. FiPy solution is implicit in time, so timestep should be 1 day.

hydrology_error_count = 0

def theta_from_zeta(z, s1, s2, b):
    theta = np.exp(s1)/s2 * (np.exp(s2*z) - np.exp(s2*b))
    return theta

N_PARAMS = 4

SENSOR_MEASUREMENT_ERR = 0.05 # metres. Theoretically, 1mm
 
def log_likelihood(params):
    
    global hydrology_error_count
    
    s1 = params[0]; s2 = params[1]
    
    log_like = 0 # result from this function. Will sum over all transects.
    
    # print(f'parameters: {params}')
    
    for _, row in transect_selected_dates.iterrows():
        # print(f'Starting with transect: {transect_name}')
        transect_name = row['transect_name']
        date0 = row['selected_period_begin']
        date1 = row['selected_period_end']
        df = dfs_by_transects[transect_name].loc[date0:date1]
        ti = transect_info[transect_info['transect_name']==transect_name]
        sensor_locations = (float(ti['sensor0_location']), float(ti['sensor1_location']))
        peat_depth = float(ti['peat_depth'])
        transect_length = float(ti['transect_length'])
        surface_elev = [float(ti['surface_elev_sensor0']), float(ti['surface_elev_sensor1'])]
        df.loc[:,'sensor_0'] += surface_elev[0]
        df.loc[:,'sensor_1'] += surface_elev[1]
        
        
        nx = int(transect_length/dx)
        
        sensor_column_names = [name for name in df.columns if 'sensor' in name]
        measurements = df[sensor_column_names]
        
        # Interpolation of DEM ele and of initial WTD        
        ele_interp = interpolate.interp1d(x=[0, transect_length], y=surface_elev, kind='linear')
        ele = ele_interp(np.arange(0, nx*dx, dx))
        b = peat_depth + ele.min() - ele
        
        sensor_WTD_ini = measurements.to_numpy()[0]
        sensor_zeta_ini = [sensor_WTD_ini[pos] for pos,value in enumerate(sensor_locations)]
        zeta_ini_interp = interpolate.interp1d(x=[0, transect_length], y=sensor_zeta_ini,kind='linear')
        zeta_ini = zeta_ini_interp(np.arange(0, nx*dx, dx))
        theta_ini = theta_from_zeta(zeta_ini, s1, s2, b)

        ndays = measurements.shape[0] - 1 # first day is ini cond
        precip = df['P'].to_numpy()
        # No precipitation in this version of the calibration!
        precip = np.zeros(shape=precip.shape)
        evapotra = df['ET'].to_numpy()
        
        if len(sensor_column_names) == 2: # P0xx transects
            zeta_boundary_values_left = measurements['sensor_0'].to_numpy()[1:] # 1st value is ini cond
            theta_boundary_values_left = theta_from_zeta(zeta_boundary_values_left, s1, s2, b[0])
            theta_boundary_values_right = None
            zeta_test_measurements = measurements['sensor_1'].to_numpy()[1:]
            
        elif len(sensor_column_names) > 2: # DOSAN and DAYUN sensors
            last_sensor = len(sensor_column_names)
            last_sensor_name = 'sensor_' + str(last_sensor) 
            zeta_boundary_values_left = measurements['sensor_0'].to_numpy()[1:]
            theta_boundary_values_left = theta_from_zeta(zeta_boundary_values_left, s1, s2, b[0])
            zeta_boundary_values_right = measurements[last_sensor_name].to_numpy()[1:]
            theta_boundary_values_right = theta_from_zeta(zeta_boundary_values_right, s1, s2, b[-1])

            # TODO: the following line might not be perfect
            zeta_test_measurements = measurements.drop(columns=['sensor_0', last_sensor_name]).to_numpy()[1:]
        
        # print('continuing...')
            
        
        try:
            with np.errstate(all='raise'):
                # simulated_wtd = hydro_calibration.hydro_1d_fipy(theta_ini, nx, dx, dt, params, ndays, sensor_locations,
                #                                             theta_boundary_values_left, theta_boundary_values_right, precip, evapotra, ele_interp, peat_depth)
                
                   simulated_wtd= hydro_calibration.hydro_1d_half_fortran(theta_ini, nx-1, dx, dt, params, ndays, sensor_locations,
                                                                          theta_boundary_values_left, theta_boundary_values_right, precip, evapotra, ele_interp, peat_depth)
            
        except: # if error in hydro computation
            hydrology_error_count += 1
            print( f"###### ERROR {hydrology_error_count} IN HYDRO #######")
            return -np.inf
        else:
            # print('#### SUCCESS!')
            sigma2 = SENSOR_MEASUREMENT_ERR ** 2
            log_like += -0.5 * np.sum((zeta_test_measurements - simulated_wtd) ** 2 / sigma2 +
                                      np.log(sigma2))
            print(f'\n SUCCESS! params = {params}')
            # print(">>>>>> SIMULATED WTD = ", simulated_wtd)
            # print(">>>>>> ZETA_MEASUREMENTS = ", zeta_test_measurements)
            # print(">>>>>> CALIBRATION SUCCESSFUL. LOG_LIKELIHOOD = ", log_like)
    
    return log_like
 
def log_prior(params):
    s1 = params[0]; s2 = params[1] 
    t1 = params[2]; t2 = params[3]
    
    # uniform priors everywhere.
    if -1000<t1<1000 and -0.1<t2<1000  and -1000<s1<1000 and -0.1<s2<1000: 
        return 0.0
    return -np.inf        
 
def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    if not np.isfinite(ll).any(): # if Error in hydro computation
        return -np.inf
    return lp + ll
 
def gen_positions_for_walkers(n_walkers, n_params):
      # Generate based on true values + noise. TODO: change in the future!
    ini_values =  [5.02738472,  0.79876819, 33.43180506, 15.04192401] # s1, s2, t1, t2
    true_values = np.array([ini_values,]*n_walkers)
    noise = (np.random.rand(n_walkers, n_params) -0.5)*20.0 # random numbers in (-10.0, + 10.0)
    return true_values + noise
 
if N_CPU > 1:  
    with Pool(N_CPU) as pool:
            
        pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
       
        nwalkers, ndim = pos.shape
         
        # save chain to HDF5 file

        fname = "mcmc_result_chain.h5"
        backend = emcee.backends.HDFBackend(fname)
        backend.reset(nwalkers, ndim) # commenting this line: continue from stored markovchain
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                        backend=backend)
        sampler.run_mcmc(pos, MCMC_STEPS, progress=True)
         
elif N_CPU == 1: # single processor
    pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
    nwalkers, ndim = pos.shape
     
    # save chain to HDF5 file
    fname = "mcmc_result_chain.h5"
    backend = emcee.backends.HDFBackend(fname)
    backend.reset(nwalkers, ndim)
     
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
                                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
    sampler.run_mcmc(pos, MCMC_STEPS, progress=True);

 
