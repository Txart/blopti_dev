# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:45:31 2020

@author: 03125327
"""
import fipy as fp
from fipy.tools import numerix
import numpy as np
import pandas as pd

def hydro_1d(theta_ini, nx, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele):
    
    mesh = fp.Grid1D(nx=nx, dx=dx)
    
    s0 = params[0]; s1 = params[1] 
    t0 = params[2]; t1 = params[3]; t2 = params[4];
    
    P = precip[0]; ET = evapotra[0]
    
    theta = fp.CellVariable(name="theta", mesh=mesh, value=theta_ini, hasOld=True)
    
    # Choice of parameterization
    # This is the underlying transimissivity: T = t0 * exp(t1 + t2*h)
    # This is the underlying storage coeff: S = s1 * exp(s0 + s1 * h) # and S_theta = s1 * theta
    # S is hidden in change from theta to h
    D = t0/s1 * numerix.exp(t1 - t2*s0/s1) * numerix.power(theta, t2/s1 - 1.)
    
    if np.isnan(D.value).any():
        raise ValueError('D is NaN')
    
    # Boussinesq eq. for theta
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + P - ET
    
    WTD_from_theta_sol = [] # returned quantity
    
    MAX_SWEEPS = 100
    
    for day in range(ndays):
        
        theta.updateOld()
        
        # BC and Source/sink update
        theta_left = numerix.exp(s0 + s1*boundary_values_left[day]) # left BC is always Dirichlet
        theta.constrain(theta_left, where=mesh.facesLeft)
        if boundary_values_right == None: # Pxx sensors. Neuman BC on the right
            theta.faceGrad.constrain(0. * mesh.faceNormals, where=mesh.facesRight)
        else:
            theta_right = numerix.exp(s0 + s1*boundary_values_right[day])
            theta.constrain(theta_right, where=mesh.facesRight)
            
        P = precip[day]; ET = evapotra[day]   
        
        res = 0.0
        for r in range(MAX_SWEEPS):
            resOld=res
            res = eq.sweep(var=theta, dt=dt)
            if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
        
        # Append to list
        theta_sol = theta.value
        theta_sol_sensors = np.array([theta_sol[sl-1] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the error
        ele_sensors = np.array([ele[sl-1] for sl in sensor_loc[1:]])
        h_sol_sensors = (np.log(theta_sol_sensors) -s0)/s1
        WTD_from_theta_sol.append(h_sol_sensors - ele_sensors) # WTD = -(ele - h)
        
    return np.array(WTD_from_theta_sol)

def fabricate_data(nx, dt, params, HINI, NDAYS, SENSOR_LOCATIONS, MAX_HEAD_BOUNDARIES=5., MAX_SOURCE=3., filename="fabricated_data.txt"):

    """
    Fabricate some toy data to test MCMC
    
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
    
    s0 = params[0]; s1 = params[1] 
    t0 = params[2]; t1 = params[3]; t2 = params[4]
    
    theta_ini = np.exp(s0 + s1*HINI) 
    
    # BC, source/sink
    boundary_sensors = np.random.rand(NDAYS, 2) * MAX_HEAD_BOUNDARIES
    PRECIPITATION = np.random.rand(NDAYS) * MAX_SOURCE
    EVAPOTRANSPIRATION = np.random.rand(NDAYS) * MAX_SOURCE
    
    wtd =hydro_1d(nx, dx, dt, params, theta_ini, NDAYS, SENSOR_LOCATIONS, boundary_sensors, PRECIPITATION, EVAPOTRANSPIRATION)

    with open(filename, 'w') as out:
        out.write('bcleft   sensor0   sensor1   sensor2   sensor3   bcright   day   P   ET\n')
        for day, l in enumerate(wtd):
            line = "   ".join([str(s_v) for s_v in l])
            line = f"{boundary_sensors[day,0]}   " + line + f"   {boundary_sensors[day,1]}   {day}   {PRECIPITATION[day]}   {EVAPOTRANSPIRATION[day]}"
            out.write( line + '\n')
    
    return 0


def read_sensors(filename):
    """
    Read sensor from file created by fabricated data

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    bcleft : TYPE
        DESCRIPTION.
    bcright : TYPE
        DESCRIPTION.
    sensor_measurements : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.
    ET : TYPE
        DESCRIPTION.

    """
    with open(filename, 'r') as f:
        df_sensors = pd.read_csv(filename, engine='python', sep='   ')
    
    bcleft = df_sensors['bcleft'].to_numpy()
    bcright = df_sensors['bcright'].to_numpy()
    sensor_measurements = df_sensors.loc[:,'sensor0':'sensor3'].to_numpy()
    day = df_sensors['day'].to_numpy()
    P = df_sensors['P'].to_numpy()
    ET = df_sensors['ET'].to_numpy()
    
    return bcleft, bcright, sensor_measurements, day, P, ET
