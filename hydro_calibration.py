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

def hydro_1d_half_fortran(theta_ini, nx, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele_interp, peat_depth):
    import fd # own fortran functions
    
    REL_TOLERANCE = 1e-5
    ABS_TOLERANCE = 1e-5
    
    s1 = params[0]; s2 = params[1] 
    t1 = params[2]; t2 = params[3]
    
    ele = ele_interp(np.linspace(0, (nx+1)*dx, nx+1))
    b = peat_depth + ele.min() - ele
    
    v_ini = theta_ini[:]  
    v = v_ini[:]
    v_old = v_ini[:] # in the previous timestep

    # Relaxation parameter
    weight = 0.1
    
    # J, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)   
    
    MAX_INTERNAL_NITER = 10000 # max niters to solve nonlinear algebraic eq of Newton's method
    
    theta_sol_list = [] # returned quantity
    
    for day in range(ndays):
        # BC and Source/sink update
        theta_left = boundary_values_left[day] # left BC is always Dirichlet. No-flux in the right all the time
        source = precip[day] - evapotra[day]
        
        # Compute tolerance. Each day, a new tolerance because source changes
        _, F = fd.j_and_f(n=nx, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=theta_left, s1=s1, s2=s2, t1=t1, t2=t2, source=source)
        rel_tol = REL_TOLERANCE * np.linalg.norm(F)

        for i in range(0, MAX_INTERNAL_NITER):
            J, F = fd.j_and_f(n=nx, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=theta_left, s1=s1, s2=s2, t1=t1, t2=t2, source=source)       
            eps_x = np.linalg.solve(J,-F)
            v = v + weight*eps_x
    
            # stopping criterion
            residue = np.linalg.norm(F) - rel_tol
            if residue < ABS_TOLERANCE:
                break
        
        # Early stopping criterion: theta cannot be negative
        if np.any(v < 0) or np.any(np.isnan(v)):
            
            raise ValueError('NEGATIVE V FOUND, ABORTING')
        
        print(f'\n Number of run internal iterations: {i}')
        v_old = v[:]
        
        # Append to list
        theta_sol = v[:]
        theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the fitness estimation
        
        theta_sol_list.append(theta_sol_sensors[0])
    
    b_sensors = np.array([b[sl] for sl in sensor_loc[1:]])
    
    zeta_from_theta_sol_sensors = zeta_from_theta(np.array(theta_sol_list), b_sensors, s1, s2)
        
    return zeta_from_theta_sol_sensors


def zeta_from_theta(x, b, s1, s2):
        return np.log(np.exp(s2*b) + s2*np.exp(-s1)*x) / s2



def hydro_1d_fipy(theta_ini, nx, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele_interp, peat_depth):

    def zeta_from_theta(x, b):
        return np.log(np.exp(s2*b) + s2*np.exp(-s1)*x) / s2
    
    mesh = fp.Grid1D(nx=nx, dx=dx)
    
    ele = ele_interp(mesh.cellCenters.value[0])
    b = peat_depth + ele.min() - ele
    
    s1 = params[0]; s2 = params[1] 
    t1 = params[2]; t2 = params[3]
    
    source = precip[0] - evapotra[0]
    
    theta = fp.CellVariable(name="theta", mesh=mesh, value=theta_ini, hasOld=True)
    
    # Choice of parameterization
    # This is the underlying conductivity: K = exp(t1 + t2*zeta). The 
    # transmissivity is derived from this and written in terms of theta
    # This is the underlying storage coeff: S = exp(s1 + s2*zeta)
    # S is hidden in change from theta to h
    # D = (numerix.exp(t1)/t2 * (numerix.power(s2 * numerix.exp(-s1) * theta + numerix.exp(s2*b), t2/s2) - numerix.exp(t2*b))) * np.power(s2 * (theta + numerix.exp(s1 + s2*b)/s2), -1)
    
    # Boussinesq eq. for theta
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=(numerix.exp(t1)/t2 * (numerix.power(s2 * numerix.exp(-s1) * theta 
                                                                                           + numerix.exp(s2*b), t2/s2) - numerix.exp(t2*b)))
                                                * np.power(s2 * (theta + numerix.exp(s1 + s2*b)/s2), -1)) + source
    
    theta_sol_list = [] # returned quantity
    
    MAX_SWEEPS = 10000
    
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
            
        source = precip[day] - evapotra[day]
        
        res = 0.0
        for r in range(MAX_SWEEPS):
            resOld=res
            res = eq.sweep(var=theta, dt=dt)
            if abs(res - resOld) < 1e-7: break # it has reached the solution of the linear system
        
        if r==MAX_SWEEPS:
            raise ValueError('Solution not converging after maximum number of sweeps')
        
        # Append to list
        theta_sol = theta.value
        theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the fitness estimation
        
        theta_sol_list.append(theta_sol_sensors[0]) 
    
    b_sensors = np.array([b[sl] for sl in sensor_loc[1:]])    
    zeta_from_theta_sol_sensors = zeta_from_theta(np.array(theta_sol_list), b_sensors)
    # print(zeta_from_theta_sol_sensors)
        
    return zeta_from_theta_sol_sensors



def hydro_1d_chebyshev(theta_ini, N, dx, dt, params, ndays, sensor_loc,
             boundary_values_left, boundary_values_right, precip, evapotra, ele_interp, PEAT_DEPTH):
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
    PEAT_DEPTH: float
        depth of impermeable bottom from lowest point of peat surface (m)
        Negative downwards.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    s1 = params[0]; s2 = params[1]
    t1 = params[2]; t2 = params[3]
    
    # Nonlinear heat equation Using Chebyshev:
    #    du/dt = d/dx(A(u)*du/dx) + S is equivalent to solving
    #    du/dt = A * u'' + dA/du * u'^2 + S
    #    BC: u'(-1) = 0; u(1) = 0; IC: u = h(0)
    # Here u is same as theta above, i.e., volumetric water content
    
  
    # IC
    v = theta_ini[:]
    
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
    b_cheby = PEAT_DEPTH + ele_cheby.min() - ele_cheby
      
    def S(u, b):
        return s2 * (u + np.exp(s1 + s2*b)/s2)
    
    def T(u, b):
        return np.exp(t1)/t2 * (np.power(s2 * np.exp(-s1) * u + np.exp(s2*b), t2/s2) - np.exp(t2*b))
    
    def dif(u, b):
        # Diffusivity
        return T(u, b) * np.power(S(u, b), -1)
    
    def dif_prime(u, b):
        # Derivative of diffusivity with respect to theta
        # Have to hardcode the derivative
        T_prime = np.exp(t1-s1) * np.power(s2/np.exp(s1)* u + np.exp(s2*b), (t2-s2)/s2)
        # S_prime = s2

        diffusivity_prime = (S(u, b) * T_prime - 
                             T(u, b) * s2) * np.power(S(u, b), -2)        
        
        return diffusivity_prime
    
    def zeta_from_theta(x, b):
        return np.log(np.exp(s2*b) + s2*np.exp(-s1)*x) / s2
    
   
    def rhs(u):
        # RHS of the PDE: du/dt = rhs(u)
        return dif_prime(u, b_cheby) * (D @ u)**2 + dif(u, b_cheby) * D2 @ u + source
    
    def forward_Euler(v, dt):
        return v + dt*rhs(v)
    
    def RK4(v, dt):
        # 4th order Runge-Kutta
                       
        # Diri BC have to be specified every time the rhs is evaluated!
        k1 = rhs(v); k1[-1] = 0 # BC
        k2 = rhs(v + dt/2*k1); k2[-1] = 0 # BC
        k3 = rhs(v + dt/2*k2); k3[-1] = 0 # BC
        k4 = rhs(v + dt*k3); k4[-1] = 0 # BC
        
        return v + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
     
    WTD_from_theta_sol = [] # returned quantity    
    
    EXPLICIT = False
    if EXPLICIT:
        # Solve iteratively
        internal_niter = int(1/dt)
    
        for day in range(ndays):
            # Update source term 
            source = precip[day] - evapotra[day]
            # Update BC
            v[-1] = boundary_values_left[day] # left BC is always Dirichlet
            
            # solve dt forward in time
            for i in range(internal_niter):
                
                v_new = forward_Euler(v, dt)
                # v_new = RK4(v, dt, params)
    
                # Reset BC
                v_new[-1] = boundary_values_left[day] # Diri
                nbc =  D[0,1:] @ v_new[1:] # No flux Neumann BC
                flux = 0.
                v_new[0] = 1/D[0,0] * (flux - nbc)
            
                v = v_new
        
            theta_sol = v_new[:][::-1] # We've got to reverse because of chebyshev transform!
            
            # Append result in terms of zeta
            theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the error
            b_cheby_sensors = np.array([b_cheby[::-1][sl] for sl in sensor_loc[1:]])
            zeta_sol_sensors = zeta_from_theta(theta_sol_sensors, s1, s2, b_cheby_sensors)
        
            WTD_from_theta_sol.append(zeta_sol_sensors[0])
    
    
    
    """
        Implicit solution
    """
    def dif_prime_prime(u, b):
        # S_prime = s2; S_prime_prime = 0., so some terms are directly zero and thus not written
        T_prime = np.exp(t1-s1) * np.power(s2/np.exp(s1)* u + np.exp(s2*b), (t2-s2)/s2)
        T_prime_prime = np.exp(t1-2*s1) * (t2 - s2) * np.power(s2/np.exp(s1) * u + np.exp(s2*b), (t2-2*s2)/s2)
        
        return (1/S(u, b))**3 * (S(u,b)**2 * T_prime_prime - 2* s2 * T_prime * S(u,b) + 2*s2**2 * T(u,b))
        
    def F(u, u_0, dt):
            # u_0 = u in previous timestep
            # dt: timestep size
            return dt * rhs(u) + u_0 - u
        
    if not EXPLICIT: # Implicit backwards Euler!
        dt = 1
        max_internal_niter = 10
        rel_tolerance = 1e-5
        abs_tolerance = 1e-5
        weight = 0.01 # relaxation parameter. weight=1 returns original problem without relaxation.
        
        for day in range(ndays):
            # Update source term 
            source = precip[day] - evapotra[day]
            # Update BC
            v[-1] = boundary_values_left[day] # left BC is always Dirichlet
            # Compute tolerance. Each day, a new tolerance because source changes
            rel_tol = rel_tolerance * np.linalg.norm(dt*rhs(theta_ini))              
            
            for i in range(0, int(max_internal_niter)):
                # solve iteratively linear eq of the form A*x+b = 0
                B = dt*(dif_prime(v, b_cheby) * (D @ v)**2 + dif(v, b_cheby) * D2 @ v) + theta_ini - v
                A = dt*(2*dif_prime(v, b_cheby) * (D @ v) * D + dif(v, b_cheby) * D2 +
                        np.eye(N+1) * (dif_prime_prime(v, b_cheby) * (D @ v)**2 +
                                       dif_prime(v, b_cheby) * D2 @ v))
                dv = np.linalg.solve(A, B)
                
                #update
                v = v + weight * dv
                
                # Set BC
                v[-1] = boundary_values_left[day] # Diri
                nbc =  D[0,1:] @ v[1:] # No flux Neumann BC
                flux = 0.
                v[0] = 1/D[0,0] * (flux - nbc)
                
                # stopping criterion
                residue = np.linalg.norm(F(v, theta_ini, dt)) - rel_tol
                print(f'res = {residue}')
                print(f'v = {v}')
                if residue < abs_tolerance:
                    print('residue smaller than tolerance!')
                    break
                
            theta_sol = v[:][::-1] # We've got to reverse because of chebyshev transform!
            # Append result in terms of zeta
            theta_sol_sensors = np.array([theta_sol[sl] for sl in sensor_loc[1:]]) # canal sensor is part of the model; cannot be part of the error
            b_cheby_sensors = np.array([b_cheby[::-1][sl] for sl in sensor_loc[1:]])
            zeta_sol_sensors = zeta_from_theta(theta_sol_sensors, b_cheby_sensors)
    
            WTD_from_theta_sol.append(zeta_sol_sensors[0])
    
    
        
          
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


