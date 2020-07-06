# -*- coding: utf-   -*-
"""
Created on Thu Mar 26 15:26:14 2020

@author: 03125327
"""
import os
import argparse
import preprocess_data, hydro_utils, utilities, hydro
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from pathlib import Path


#%%
"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run MCMC parameter estimation')

parser.add_argument('--ncpu', default=10, help='(int) Numer of processors', type=int)
parser.add_argument('-cl','--chainlength', default=10, help='(int) Length of MCMC chain', type=int)
parser.add_argument('-w','--nwalkers', default=32, help='(int) Number of walkers in parameter space', type=int)
args = parser.parse_args()

N_CPU = args.ncpu
MCMC_STEPS = args.chainlength
N_WALKERS = args.nwalkers

N_PARAMS = 6

#%% get wtd and P sensor data
import get_data
relative_path_datafolder = './data'
absolute_path_datafolder = os.path.abspath(relative_path_datafolder)

# df_wt = get_data.get_wt_data() # fix this

P, ET = get_data.get_historic_P_ET(absolute_path_datafolder)

# Narrow down to 6 sensors. This will have to be much more elaborated in the future
N_MEASUREMENTS = 6

# wtd_data = df_wt.iloc[0:N_MEASUREMENTS].measurement.to_numpy() * .001 # To meters
wtd_data = -np.random.rand(N_MEASUREMENTS) # testing
wtd_data_err = 1.0 # Ask Roy, Imam?

P = P[0:N_MEASUREMENTS]; ET = ET[0:N_MEASUREMENTS]


#%% get initial WTD for the model
filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)

dem_rst_fn = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
can_rst_fn = Path(filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
peat_depth_rst_fn = Path(filenames_df[filenames_df.Content == 'peat_depth_raster'].Path.values[0])
params_fn = Path(filenames_df[filenames_df.Content == 'parameters'].Path.values[0])
WTD_folder = Path(filenames_df[filenames_df.Content == 'WTD_input_and_output_folder'].Path.values[0])
weather_fn = Path(filenames_df[filenames_df.Content == 'historic_precipitation'].Path.values[0])
# Choose smaller study area
STUDY_AREA = (200,300), (150,220)
wtd_old_fn = dem_rst_fn

_, wtd_old , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(STUDY_AREA, wtd_old_fn, can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn)

#TODO. RIGHT NOW, RANDOM
SENSOR_POSITIONS = [(np.random.randint(low=0, high=STUDY_AREA[0][1]-STUDY_AREA[0][0]), np.random.randint(low=0, high=STUDY_AREA[1][1]-STUDY_AREA[1][0])) for i in range(N_MEASUREMENTS)]


#%%
# Prepare hydrology to run

DAYS = 1
N_BLOCKS = 0


# Generate adjacency matrix, and dictionary. Need to do this every time?
CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(STUDY_AREA, can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)

# Read parameters
PARAMS_df = preprocess_data.read_params(params_fn)
CANAL_WATER_LEVEL = PARAMS_df.canal_water_level[0]
DIRI_BC = PARAMS_df.diri_bc[0]; HINI = PARAMS_df.hini[0];
# ET = np.array([PARAMS_df.ET[0]])
TIMESTEP = PARAMS_df.timeStep[0]

print(">>>>> WARNING, OVERWRITING PEAT DEPTH")
peat_depth_arr[peat_depth_arr < 2.] = 2.

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_masked = peat_type_arr * catchment_mask
peat_bottom = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!
#

srfcanlist =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)


# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, 0, [], CNM)

ny, nx = dem.shape
dx = 1.; dy = 1. # metres per pixel  (Actually, pixel size is 100m x 100m, so all units have to be converted afterwards)

boundary_arr = boundary_mask * (dem - DIRI_BC) # constant Dirichlet value in the boundaries

ele = dem * catchment_mask


phi_ini = ele + wtd_old #initial h (gwl) in the compartment.
phi_ini = phi_ini * catchment_mask
       
wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
for canaln, coords in enumerate(c_to_r_list):
    if canaln == 0: 
        continue # because c_to_r_list begins at 1
    wt_canal_arr[coords] = wt_canals[canaln] 

#%%
# Transimsivity and storativity parameterization
# def tra_sto_old(kadjust):
#     """
#     First trial when the only parameter was kadjust.
#     """
    
    
#     h_to_tra_and_C_dict, _ = hydro_utils.peat_map_interp_functions(Kadjust=kadjust) # Load peatmap soil types' physical properties dictionary
    
#     tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_masked,
#                                                 gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
#     sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_masked,
#                                                 gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
#     sto_to_cut = sto_to_cut * catchment_mask.ravel()

#     return h_to_tra_and_C_dict, tra_to_cut, sto_to_cut


def transmissivity(h, t0, t1, t2, t_sapric_coef, peat_type_masked=peat_type_masked):
    
    hemic_multiplier =  1.0 * np.logical_and(peat_type_masked > 0, peat_type_masked < 5)
    sapric_multiplier = t_sapric_coef * (peat_type_masked >6)
    type_multiplier = hemic_multiplier + sapric_multiplier
    
    return type_multiplier * t0 * np.exp(t1 * (-h)**t2)

def storage(h, s0, s1, s2, s_sapric_coef, peat_type_masked=peat_type_masked):
    
    hemic_multiplier =  1.0 * np.logical_and(peat_type_masked > 0, peat_type_masked < 5)
    sapric_multiplier = s_sapric_coef * (peat_type_masked >6)
    type_multiplier = hemic_multiplier + sapric_multiplier
    
    return type_multiplier * (s0 - s1 * (-h)**s2)


#%%
# Bayesian parameter estimation

def log_likelihood(theta):
    t0 = theta[0]; t1 = theta[1]; t2 = theta[2];
    s0 = theta[3]; s1 = theta[4]; s2 = theta[5]

    wtd_sensors = hydro.hydrology('transient', SENSOR_POSITIONS, nx, ny, dx, dy, DAYS, ele, phi_ini,
                      catchment_mask, wt_canal_arr, boundary_arr,
                      peat_type_mask=peat_type_masked, peat_bottom_arr=peat_bottom, 
                      transmissivity = transmissivity, t0=t0, t1=t1, t2=t2, t_sapric_coef=1.0,
                      storage=storage, s0=s0, s1=s1, s2=s2, s_sapric_coef=1.0,
                      diri_bc=DIRI_BC, plotOpt=False, remove_ponding_water=True,
                      P=P, ET=ET, dt=TIMESTEP)
    sigma2 = wtd_data_err ** 2
    return -0.5 * np.sum((wtd_data - wtd_sensors) ** 2 / sigma2 + np.log(sigma2))

# maximum likelihood optimization
opt_ML = False
if opt_ML:
    from scipy.optimize import minimize
    MAXITER = 5
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.random.randn(1)
    soln = minimize(nll, initial, args=(wtd_data, wtd_data_err), options={'maxiter':MAXITER, 'disp':True})
    kadj_ml = soln.x
    
    print("Maximum likelihood estimates:")
    print("k_adjust = {0:.3f}".format(kadj_ml))


def log_prior(theta):
    t0 = theta[0]; t1 = theta[1]; t2 = theta[2];
    s0 = theta[3]; s1 = theta[4]; s2 = theta[5]
    # uniform priors everywhere.
    if 0<t0<2 and 0<t1<2 and 0<t2<2  and 0<s0<2 and 0<s1<2 and 0<s2<2: 
        return 0.0
    
    return -np.inf

def log_probability(theta):

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

import emcee


with Pool(N_CPU) as pool:
    pos = (np.random.rand(N_WALKERS,N_PARAMS) + 0.05) * 1.
    
    nwalkers, ndim = pos.shape
    
    # save chain to HDF5 file
    fname = "output/mcmc_result_chain.h5"
    backend = emcee.backends.HDFBackend(fname)
    backend.reset(nwalkers, ndim)
    
    sampler_multi = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
    sampler_multi.run_mcmc(pos, MCMC_STEPS, progress=True)

