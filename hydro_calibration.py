# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 08:45:31 2020

@author: 03125327
"""
import fipy as fp
from fipy.tools import numerix
import numpy as np
import pandas as pd

#%%
def hydro_1d_fipy(theta_ini, nx, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele):
    # TODO: implement zeta and new parameterization
    raise NotImplementedError("Not imiplemented yet!")
    
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
    
    MAX_SWEEPS = 1000
    
    for day in range(ndays):
        
        theta.updateOld()
        
        # BC and Source/sink update
        theta_left = boundary_values_left[day] # left BC is always Dirichlet
        theta.constrain(theta_left, where=mesh.facesLeft)
        if boundary_values_right == None: # Pxx sensors. Neuman BC on the right
            theta.faceGrad.constrain(0. * mesh.faceNormals, where=mesh.facesRight)
        else:
            theta_right = boundary_values_right[day]
            theta.constrain(theta_right, where=mesh.facesRight)
            
        P = precip[day]; ET = evapotra[day]   
        
        res = 0.0
        for r in range(MAX_SWEEPS):
            resOld=res
            res = eq.sweep(var=theta, dt=dt)
            if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
        
        # Append to list
        theta_sol = theta.value
        theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the fitness estimation
        h_sol_sensors = (np.log(theta_sol_sensors) -s0)/s1

        WTD_from_theta_sol.append(h_sol_sensors[0]) 
        
    return np.array(WTD_from_theta_sol)

def hydro_1d_chebyshev(theta_ini, N, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele_interp, bottom):
    """
    Returns zeta = -(ele - h) to compare directly with sensor values.
    
    Parameters
    ----------
    theta_ini : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    ndays : TYPE
        DESCRIPTION.
    sensor_loc : TYPE
        DESCRIPTION.
    boundary_values_left : TYPE
        DESCRIPTION.
    boundary_values_right : TYPE
        DESCRIPTION.
    precip : TYPE
        DESCRIPTION.
    evapotra : TYPE
        DESCRIPTION.
    ele_interp : scipy interpolation function
        We need to pass the interpolation function bc Chebyshev transforms 
        to cos(x) space
    bottom: float
        depth of impermeable bottom below ref level z=0. In m. Negative downwards.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    s1 = params[0]; s2 = params[1];
    t1 = params[2]; t2 = params[3]
    
    # Nonlinear heat equation Using Chebyshev:
    #    du/dt = d/dx(A(u)*du/dx) + S is equivalent to solving
    #    du/dt = A * u'' + dA/du * u'^2 + S
    #    BC: u'(-1) = 0; u(1) = 0; IC: u = h(0)
    # Here u is same as theta above, i.e., volumetric water content
    
  
    # IC
    v_old = theta_ini[:]
    
    if len(theta_ini) != N+1: #chebyshev adds one point to the mesh
        raise ValueError("initial value not same size as discretization")
    
    D,x = cheb(N)
    # domain of actual x is [0, N*dx]
    # In order to map it to [-L, L] = [-N*dx/2, +N*dx/2] in cheby space,
    # we need to rescale accordingly:
    L = N*dx/2
    x = L * x; D = D/L
    D2 = D @ D # matrix multiplication
    ele_cheby = ele_interp((x+L)[::-1])[::-1]
        
    def dif(u, params):
        # Dffusivity
        s1 = params[0]; s2 = params[1];
        t1 = params[2]; t2 = params[3]
        
        #diffusivity = t0/s1 * np.exp(t1 - t2*s0/s1) * np.power(u, t2/s1 - 1.) # Old version
        C = np.exp(s1+s2*(bottom - ele_cheby))/s2
        diffusivity = np.exp(t1 + t2*ele_cheby) / (t2*s2) * 1/(u + C) * (
            np.exp(-s1*t2/s2) * (s2*u + s2*C)**(t2/s2) - np.exp(t2*(bottom-ele_cheby)))
        
        return diffusivity
    
    def dif_prime(u, params):
        # Derivative of diffusivity with respect to theta
        # Have to hardcode the derivative
        s1 = params[0]; s2 = params[1];
        t1 = params[2]; t2 = params[3];
        
        C = np.exp(s1+s2*(bottom - ele_cheby))/s2
        # diffusivity_prime = t0/s1**2 * (t2-s1) * np.exp(t1 - t2*s0/s1) * np.power(u, (t2 - 2*s1)/s1)
        diffusivity_prime = np.exp(t1+t2*ele_cheby)/(t2*s2*(u + C)**2) * (
            np.exp(-s1*t2/s2)*s2**(t2/s2) * (t2-s2)/s2 * (u + C)**(t2/s2) +
            np.exp(t2*(bottom - ele_cheby)))
        
        
        return diffusivity_prime
    
    def zeta_from_theta(x, s1, s2, ele_cheby):
        C = np.exp(s1+s2*(bottom - ele_cheby))/s2
        return (np.log(s2*(x+C)) -s1) / s2
    
    # TODO: REMOVE IN THE FUTURE; ONLY D NEEDED
    def S(u, params):
        s1 = params[0]; s2 = params[1];
        C = np.exp(s1+s2*(bottom - ele_cheby))/s2
        return s2*(u + C)
    
    def T(u, params):
        s1 = params[0]; s2 = params[1];
        t1 = params[2]; t2 = params[3];
        C = np.exp(s1+s2*(bottom - ele_cheby))/s2
        return np.exp(t1+t2*ele_cheby)/t2 * (np.exp(-s1*t2/s2) * (s2*u + s2*C)**(t2/s2) -
                                             np.exp(t2*(bottom-ele_cheby)))
    
   
    def rhs(u, params):
        # RHS of the PDE: du/dt = rhs(u)
        return dif_prime(u, params) * (D @ u)**2 + dif(u, params) * D2 @ u + source
    
    def forward_Euler(v_old, dt, params):
        return v_old + dt*rhs(v_old, params)
    
    def RK4(v_old, dt, params):
        # 4th order Runge-Kutta
                       
        # Diri BC have to be specified every time the rhs is evaluated!
        k1 = rhs(v_old, params); k1[-1] = 0 # BC
        k2 = rhs(v_old + dt/2*k1, params); k2[-1] = 0 # BC
        k3 = rhs(v_old + dt/2*k2, params); k3[-1] = 0 # BC
        k4 = rhs(v_old + dt*k3, params); k4[-1] = 0 # BC
        
        return v_old + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
     
    WTD_from_theta_sol = [] # returned quantity    
    
    # Solve iteratively
    internal_niter = int(1/dt)

    for day in range(ndays):
        # Update source term 
        source = (precip[day] - evapotra[day]) / internal_niter
        # Update BC
        v_old[-1] = boundary_values_left[day] # left BC is always Dirichlet
        
        # solve dt forward in time
        for i in range(internal_niter):
            
            v_new = forward_Euler(v_old, dt, params)
            # v_new = RK4(v_old, dt, params)

            # Reset BC
            v_new[-1] = boundary_values_left[day] # Diri
            nbc =  D[0,1:] @ v_new[1:] # No flux Neumann BC
            flux = 0.
            v_new[0] = 1/D[0,0] * (flux - nbc)
        
            v_old = v_new
            
        # Compare with measured and append result
        theta_sol = v_new[:][::-1] # We've got to reverse because of chebyshev transform!
        
        theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the error
        ele_cheby_sensors = np.array([ele_cheby[::-1][sl] for sl in sensor_loc[1:]])
        zeta_sol_sensors = zeta_from_theta(theta_sol_sensors, s1, s2, ele_cheby_sensors)

        WTD_from_theta_sol.append(zeta_sol_sensors[0])
        
        print(f" theta = {theta_sol_sensors}")
        print(f" zeta = {zeta_sol_sensors}")
        
          
    return np.array(WTD_from_theta_sol)

#%%
def cheb(N):
    '''Chebyshev polynomial differentiation matrix.
       Ref.: https://github.com/nikola-m/another-chebpy/blob/master/chebPy.py
    '''
    x = np.cos(np.pi*np.linspace(0,N,N+1)/N)
    c=np.zeros(N+1)
    c[0]=2.
    c[1:N]=1.
    c[N]=2.
    c = c * (-1)**np.linspace(0,N,N+1)
    X = np.tile(x, (N+1,1))
    dX = X.T - X # other way around!
    D = np.dot(c.reshape(N+1,1),(1./c).reshape(1,N+1))
    D = D / (dX+np.eye(N+1))
    D = D - np.diag( D.T.sum(axis=0) )
    return D,x


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
