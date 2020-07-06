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
import pandas as pd
from scipy import interpolate
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee

"""
Definitions
"""

def fabricate_data(nx, dt, s0, s1, t0, t1, t2, HINI, NDAYS, SENSOR_LOCATIONS, MAX_HEAD_BOUNDARIES=5., MAX_SOURCE=3., filename="fabricated_data.txt"):

    """

    Parameters
    ----------
    nx : int
        Number of mesh subdivisions
    dt : float
        Time step in days
    s0, s1 : float
        params of S
    t0, t1, t2 : float
        params of T
    NDAYS : int
        Number of days
    SENSOR_LOCATIONS : list
        Location of sensors relative to mesh. Min value 0; max value (nx-1)
    MAX_HEAD_BOUNDARIES : float, optional

        Maximum head in the boundaries. The default is 5.. BC is somewhat random, that's why that maximum is there.

    MAX_SOURCE : float, optional
        Maximum value of P and ET. The default is 3..
    filename : str, optional
        Name of file where to write the fabricated data. The default is "fabricated_data.txt".

    Returns
    -------
    None. Instead, writes 
    WTD of sensor(i) (with i from 0 to len(SENSOR_LOCATIONS)); day; P; ET
    to the file specified in filename
    

    """
    dx = 1
    mesh = fp.Grid1D(nx=nx, dx=dx)
    
    h = fp.CellVariable(name="head", mesh=mesh, value=HINI, hasOld=True)

    theta = fp.CellVariable(name="theta", mesh=mesh, value=numerix.exp(s0_true + s1_true*h.value), hasOld=True)
    
    D = t0_true/s1_true * numerix.exp(t1_true* ((numerix.log(theta) -s0_true)/s1_true)**t2_true)/ theta
    
    # BC, source/sink
    boundary_sensors = np.random.rand(NDAYS, 2) * MAX_HEAD_BOUNDARIES
    PRECIPITATION = np.random.rand(NDAYS) * MAX_SOURCE
    EVAPOTRANSPIRATION = np.random.rand(NDAYS) * MAX_SOURCE
    P = PRECIPITATION[0]; ET = EVAPOTRANSPIRATION[0]
    
    # Boussinesq eq. in theta
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + P - ET
    
    MAX_SWEEPS = 100

    
    with open(filename, 'w') as out:
        out.write('sensor0   sensor1   sensor2   sensor3   day   P   ET\n')
    
    for day in range(NDAYS):
        
        theta.updateOld()
        
        # BC and Source/sink update
        theta_left = numerix.exp(s0_true + s1_true*boundary_sensors[day,0])
        theta_right = numerix.exp(s0_true + s1_true*boundary_sensors[day,1])
        theta.constrain(theta_left, where=mesh.facesLeft); theta.constrain(theta_right, where=mesh.facesRight)
        P = PRECIPITATION[day]; ET = EVAPOTRANSPIRATION[day]
        
        
        res = 0.0
        for r in range(MAX_SWEEPS):
            resOld=res
            res = eq.sweep(var=theta, dt=dt)

            if abs(res - resOld) < 1e-7: break # it has reached the solution of the linear system
      
        with open(filename, 'a') as out:
            h_from_theta = (numerix.log(theta) -s0)/s1
            sensor_values = [h_from_theta.value[loc] for loc in SENSOR_LOCATIONS]
            line = "   ".join([str(s_v) for s_v in sensor_values])
            line = line + f"   {day}   {P}   {ET}"
            out.write( line + '\n')

def read_sensors(filename):
    with open(filename, 'r') as f:
        df_sensors = pd.read_csv(filename, engine='python', sep='   ')
    
    sensor_measurements = df_sensors.loc[:,'sensor0':'sensor3'].to_numpy()
    day = df_sensors['day'].to_numpy()
    P = df_sensors['P'].to_numpy()
    ET = df_sensors['ET'].to_numpy()
    
    return sensor_measurements, day, P, ET

def hydro_1d(nx, dx, dt, params, theta_ini, ndays, sensor_loc):
    mesh = fp.Grid1D(nx=nx, dx=dx)
    
    s0 = params[0]; s1 = params[1] 
    t0 = params[2]; t1 = params[3]; t2 = params[4];
    
    P = precip[0]; ET = evapotra[0]
    
    theta = fp.CellVariable(name="theta", mesh=mesh, value=theta_ini, hasOld=True)
    
    # Choice of parameterization
    # This is the underlying transimissivity: T = t0 * exp(t1 * h**t2)
    # This is the underlying storage coeff: S = s1 * exp(s0 + s1 * h) # and S_theta = s1 * theta
    # S is hidden in change from theta to h
    D = t0/s1 * numerix.exp(t1* ((numerix.log(theta) -s0)/s1)**t2)/ theta 
    
    if np.isnan(D.value).any() or (D<0).any():
        raise ValueError('D is non-positive')
    
    # Boussinesq eq. for theta
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + P - ET
    
    h_from_theta_sol = [] # returned quantity
    
    MAX_SWEEPS = 100
    
    for day in range(ndays):
        
        theta.updateOld()
        
        # BC and Source/sink update
        boundary_sensors = [measurements[day,0], measurements[day, -1]]
        theta_left = numerix.exp(s0 + s1*boundary_sensors[0])
        theta_right = numerix.exp(s0 + s1*boundary_sensors[1])
        theta.constrain(theta_left, where=mesh.facesLeft); theta.constrain(theta_right, where=mesh.facesRight)
        P = precip[day]; ET = evapotra[day]   
        
        res = 0.0
        for r in range(MAX_SWEEPS):
            resOld=res
            res = eq.sweep(var=theta, dt=dt)
            if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
        
        # Append to list
        theta_sol = theta.value
        theta_sol_sensors = np.array([theta_sol[i] for i in sensor_loc])
        h_from_theta_sol.append((np.log(theta_sol_sensors) -s0)/s1)
    
    return np.array(h_from_theta_sol)

#%%
"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run MCMC parameter estimation')

parser.add_argument('--ncpu', default=1, help='(int) Number of processors', type=int)
parser.add_argument('-cl','--chainlength', default=20, help='(int) Length of MCMC chain', type=int)
parser.add_argument('-w','--nwalkers', default=10, help='(int) Number of walkers in parameter space', type=int)
args = parser.parse_args()

N_CPU = args.ncpu
MCMC_STEPS = args.chainlength
N_WALKERS = args.nwalkers

N_PARAMS = 5

#%%
"""
Fabricate sensor data
From synthetic fipy 1d simulation
"""
s0_true = 0.1; s1_true = 0.2
t0_true = 1.; t1_true = 0.01; t2_true = 1.1
true_params = [s0_true, s1_true, t0_true, t1_true, t2_true]

HINI = 8.

SENSOR_LOCATIONS = [0, 12, 67, 94]
NDAYS = 5
nx_fabricate=100; dt=1.


PLOT_SETUP = False
if PLOT_SETUP:
    plt.figure()
    fig, axes = plt.subplots(2)
    axS, axT = axes
    
    z = np.arange(0,10,0.01)
    axS.plot(s1_true * np.exp(s0_true + s1_true * z), z)
    axT.plot(t0_true * np.exp(t1_true * z**t2_true), z)
    axS.set_xlabel('S(h)'); axS.set_ylabel('z')
    axT.set_xlabel('T(h)'); axT.set_ylabel('z')


# Uncomment this to fabricate and rewrite some data
# fabricate_data(nx_fabricate, dt, s0_true, s1_true, t0_true, t1_true, t2_true, HINI, NDAYS, SENSOR_LOCATIONS)


#%%
"""
Get sensor data
TODO: rewrite for sensors of the same time! MCMC takes care of parallelization
"""
filename = 'fabricated_data.txt'
measurements, days, precip, evapotra = read_sensors(filename)

if PLOT_SETUP:
    plt.figure()
    for i, line in enumerate(measurements):
        plt.plot(line, label=str(i))
    plt.legend()

#%%
"""
MCMC parameter estimation
""" 
nx = 10
dx = 1.
dt = 1.


NDAYS = 5 # TODO: Get this from data?

SENSOR_MEASUREMENT_ERR = 0.05 # metres. Theoretically, 1mm

# Correct sensor positions to accommodate new nx
sensor_locations = np.array(SENSOR_LOCATIONS) * nx / nx_fabricate
sensor_locations = np.rint(sensor_locations).astype(int)

def log_likelihood(params):
    s0 = params[0]; s1 = params[1] 
    t0 = params[2]; t1 = params[3]; t2 = params[4];
    
    # IC, interpolated from initial sensor values
    hini_interp = interpolate.interp1d(SENSOR_LOCATIONS, measurements[0])
    hini = hini_interp(np.arange(0, nx, dx))
    theta_ini = np.exp(s0 + s1*hini) 
    
    try:
        simulated_wtd = hydro_1d(nx, dx, dt, params, theta_ini, NDAYS, sensor_locations)
    # TODO: this error handling might be the reason of the thing not working. Check!!
    except: # if error in hydro computation
        print("###### SOME ERROR IN HYDRO #######")
        return -np.inf # or -np.inf?
    else:
        sigma2 = SENSOR_MEASUREMENT_ERR ** 2
        return -0.5 * np.sum((measurements - simulated_wtd) ** 2 / sigma2 + np.log(sigma2))

# maximum likelihood optimization
opt_ML = False
if opt_ML:
    from scipy.optimize import minimize
    MAXITER = 100
    np.random.seed(42)
    nll = lambda x :-log_likelihood(x)
    initial = np.random.rand(5)
    soln = minimize(nll, initial, options={'maxiter':MAXITER, 'disp':True})
    kadj_ml = soln.x
    
    print("Maximum likelihood estimates:")
    print(f"k_adjust = {kadj_ml}")


def log_prior(params):
    s0 = params[0]; s1 = params[1] 
    t0 = params[2]; t1 = params[3]; t2 = params[4];
    # uniform priors everywhere.
    if -0.1<t0<1000 and -0.1<t1<10 and -0.1<t2<10  and -0.1<s0<100 and -0.1<s1<100: 
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
    true_values = np.array([[s0_true,s1_true,t0_true,t1_true,t2_true],]*n_walkers)
    noise = (np.random.rand(n_walkers, n_params) -0.5)*0.2 # random numbers in (-0.1, +0.1)
    return true_values + noise

if N_CPU > 1:
    with Pool(N_CPU) as pool:
        pos = gen_positions_for_walkers(N_WALKERS, N_PARAMS)
        
        nwalkers, ndim = pos.shape
        
        # save chain to HDF5 file
        fname = "mcmc_result_chain.h5"
        backend = emcee.backends.HDFBackend(fname)
        backend.reset(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)
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

#%%
import corner


flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
print(flat_samples.shape)
labels = ['s0', 's1', 't0', 't1', 't2']
fig = corner.corner(
    flat_samples, labels=labels, truths=true_params
);
fig.savefig("MCMC_corner_result.png")