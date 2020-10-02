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
# """
# Fabricate sensor data
# From syntheti fipy 1d simulation
# """
# s0_true = 0.1; s1_true = 0.2
# t0_true = 1.; t1_true = 0.01; t2_true = 1.1
# true_params = [s0_true, s1_true, t0_true, t1_true, t2_true]
 
# HINI = 8.
 
# SENSOR_LOCATIONS = [0, 12, 67, 99]
# NDAYS = 5
# nx_fabricate=100; dt=1.
 
 
# PLOT_SETUP = False
# if PLOT_SETUP:
#     fig, axes = plt.subplots(2)
#     axS, axT = axes   

#     z = np.arange(0,10,0.01)
#     axS.plot(s1_true * np.exp(s0_true + s1_true * z), z)
#     axT.plot(t0_true * np.exp(t1_true * z**t2_true), z)
#     axS.set_xlabel('S(h)'); axS.set_ylabel('z')
#     axT.set_xlabel('T(h)'); axT.set_ylabel('z')
 
 
# Uncomment this to fabricate and rewrite some data
# hydro_calibration.fabricate_data(nx_fabricate, dt, true_params, HINI, NDAYS, SENSOR_LOCATIONS)
 
 
#%%
# """
# Get fabricated sensor data
# TODO: rewrite for sensors of the same time! MCMC takes care of parallelization
# """
# filename = 'fabricated_data.txt'
# bcleft, bcright, measurements, days, precip, evapotra = read_sensors(filename)
# boundary = np.array(list(zip(bcleft, bcright)))
 
# if PLOT_SETUP:
#     plt.figure()
#     for i, line in enumerate(measurements):
#         plt.plot(line, label=str(i))
#     plt.legend()
 
#%%
"""
 Get data from measurements
 Choose relevant transects.
 Slice by date
""" 
fn_weather_data = Path('data/weather_station_historic_data.xlsx')
dfs_by_transects = get_data.main(fn_weather_data)

# Choose transects
relevant_transects = ['P021', 'P012']
dfs_relevant_transects = {x: dfs_by_transects[x] for x in relevant_transects}
 
dfs_sliced_relevant_transects = {}
# Slice by julian day
jday_bounds = {'P021':[660, 673], # 660: 22/10/2019; 830: 9/4/2020
               'P012':[658, 670]}

for key, df in dfs_relevant_transects.items():
    sliced_df = df.loc[jday_bounds[key][0]:jday_bounds[key][1]]
    dfs_sliced_relevant_transects[key] = sliced_df

# TODO: Maybe put all this in an excel    
# Data invented. Check from raster data
DEM_RESOLUTION = 100 # m/pixel
sensor_locations = {'P021':[0, -1],
                    'P012':[0, -1]}  # sensor locations wrt position in grid

transect_pixels = {'P021': 3,
                   'P012': 4} # length in pixels

surface_elev_pixels = {'P021':[2,4,5],
                       'P012':[1,1.4,2, 2.2]} # m above common ref point z=0

peat_depth = {'P021':-4,
              'P012':-2} # m below lowest peat surface elevation

mesh_dt = {'P021':1e-4,
           'P012':1e-4} # in days

mesh_nx = {'P021':10,
           'P012': 10}

# put all data into one meta-dictionary ordered by transect
data_dict = {}
for tran in relevant_transects:
    data_dict[tran] = {'df':dfs_sliced_relevant_transects[tran],
                       'sen_loc':sensor_locations[tran],
                       'dt':mesh_dt[tran], 'nx':mesh_nx[tran],
                       'surface_elev_pixels':surface_elev_pixels[tran],
                       'peat_depth':peat_depth[tran],
                       'transect_pixels': transect_pixels[tran]}  
    
    


#%%

"""
MCMC parameter estimation
""" 

def theta_from_zeta(z, s1, s2, b):
    theta = np.exp(s1)/s2 * (np.exp(s2*z) - np.exp(s2*b))
    return theta

N_PARAMS = 4

SENSOR_MEASUREMENT_ERR = 0.05 # metres. Theoretically, 1mm
 
def log_likelihood(params):
    
    s1 = params[0]; s2 = params[1]
    
    log_like = 0 # result from this function. Will sum over all transects.

    for transect_name, dic in data_dict.items():
        df = dic['df']
        sensor_locations = dic['sen_loc']
        dt = dic['dt']; nx = dic['nx']
        surface_elev_pixels = dic['surface_elev_pixels']
        peat_depth = dic['peat_depth']
        transect_pixels = dic['transect_pixels']
        
        transect_length = (transect_pixels - 1) * DEM_RESOLUTION
        dx = int(transect_length/nx)
        
        if transect_pixels != len(surface_elev_pixels):
            raise ValueError('DEM length must be the same as transect length ')
        
        sensor_column_names = [name for name in df.columns if 'sensor' in name]
        measurements = df[sensor_column_names]
        
        # Interpolation of DEM ele and of initial WTD        
        grid_meters = np.arange(0, len(surface_elev_pixels)*DEM_RESOLUTION, DEM_RESOLUTION)
        ele_interp = interpolate.interp1d(x=grid_meters, y=surface_elev_pixels, kind='linear')
        ele = ele_interp(np.arange(0, nx*dx, dx))
        b = peat_depth + ele.min() - ele
        
        sensor_WTD_ini = measurements.to_numpy()[0]
        sensor_zeta_ini = [sensor_WTD_ini[pos] for pos,value in enumerate(sensor_locations)]
        zeta_ini_interp = interpolate.interp1d(x=grid_meters[sensor_locations], y=sensor_zeta_ini,kind='linear')
        zeta_ini = zeta_ini_interp(np.arange(0, nx*dx, dx))
        theta_ini = theta_from_zeta(zeta_ini, s1, s2, b)

        ndays = measurements.shape[0] - 1 # first day is ini cond
        precip = df['P'].to_numpy()
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
        
        try:
        # CHEBYSHEV
            simulated_wtd = hydro_calibration.hydro_1d_chebyshev(theta_ini, nx-1, dx, dt, params, ndays, sensor_locations,
                                                        theta_boundary_values_left, theta_boundary_values_right, precip, evapotra, ele_interp, peat_depth)
            # print(simulated_wtd)
        # FIPY
            # simulated_wtd = hydro_calibration.hydro_1d_fipy(theta_ini, nx, dx, dt, params, ndays, sensor_locations,
            #                                             theta_boundary_values_left, theta_boundary_values_right, precip, evapotra, ele)
        except: # if error in hydro computation
            print("###### SOME ERROR IN HYDRO #######")
            return -np.inf
        else:
            sigma2 = SENSOR_MEASUREMENT_ERR ** 2
            log_like += -0.5 * np.sum((zeta_test_measurements - simulated_wtd) ** 2 / sigma2 +
                                      np.log(sigma2))
    
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
    ini_values = [2.0, 1.1, 1.0, 2.0] # s1, s2, t1, t2
    true_values = np.array([ini_values,]*n_walkers)
    noise = (np.random.rand(n_walkers, n_params) -0.5)*0.2 # random numbers in (-0.1, +0.1)
    return true_values + noise
 
if N_CPU > 1:
    # Turn off NumPy automatic parallelization
    
    with Pool(N_CPU) as pool:
            
        pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
       
        nwalkers, ndim = pos.shape
         
        # save chain to HDF5 file

        fname = "mcmc_result_chain.h5"
        backend = emcee.backends.HDFBackend(fname)
        # backend.reset(nwalkers, ndim) # commenting this line: continue from stored markovchain
        
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
     
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
    sampler.run_mcmc(pos, MCMC_STEPS, progress=True);

 
