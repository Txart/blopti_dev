# -*- coding: utf-8 -*-
"""
CHECK ACCURACY WITH INCREASING SPATIAL RESOLUTION
    
Created on Wed Sep 16 10:02:16 2020

@author: 03125327
"""
import time
import matplotlib.pyplot as plt
import numpy as np


#%%
import fd # own fortran functions



def solve_with_given_N(N, params):
    
    dx = 2.0/N
    
    DIRI = 0
    SOURCE = 2
    INI_VALUE = 0.2
    MAX_INTERNAL_NITER = 10000 # max niters to solve nonlinear algebraic eq of Newton's method
    
    rel_tolerance = 1e-5
    abs_tolerance = 1e-5
    
    # Relaxation parameter
    weight = 1/N
    
    dt = 1.0 # in days
    v_ini = np.ones(shape=N+1)*INI_VALUE
    v_ini[0] = DIRI
    
    v = v_ini[:]
    v_old = v_ini[:] # in the previous timestep
    
    
    b = np.ones(shape=v.shape) * (-4)
    
    s1 = params[0]; s2 = params[1]
    t1 = params[2]; t2 = params[3]

    
    c_start_time = time.time()

    # Update source
    source = SOURCE
    
    # Update BC
    DIRI = DIRI
    # No-flux in the right all the time
    
    # Compute tolerance. Each day, a new tolerance because source changes
    _, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=source)
    rel_tol = rel_tolerance * np.linalg.norm(F)
    print(rel_tol)
    
    for i in range(0, MAX_INTERNAL_NITER):
        J, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=source)
        
        eps_x = np.linalg.solve(J,-F)
        v = v + weight*eps_x

        # stopping criterion
        residue = np.linalg.norm(F) - rel_tol
        if residue < abs_tolerance:
            print(f'Solution of the Newton linear system in {i} iterations')
            break
    
    v_old = v[:]
        
    time_spent = time.time() - c_start_time
    print(f"Finite diff FORTRAN, {N} time = {time_spent}") 
    
    return v

#%%
# Run accuracy tests
Ns = [10, 25, 50, 100, 200]

N_PARAMS = 10
rnd_params = np.random.rand(N_PARAMS,4) * 2

v_sols = [[] for i in range(N_PARAMS)]

for nN, N in enumerate(Ns):
    for nparam, params in enumerate(rnd_params):
        v_sol = solve_with_given_N(N, params)
        v_sols[nparam].append(v_sol)
        

#%%
# Plot
        
cmap = plt.cm.Accent
cmaplist = [cmap(int(i)) for i in np.linspace(0,10,len(Ns))]

for nparam, params in enumerate(rnd_params):
    plt.figure(nparam, figsize=(8, 6), dpi=400)
    for nN, N in enumerate(Ns):
        x = np.linspace(0,2,N+1)
        plt.plot(x, v_sols[nparam][nN], color=cmaplist[nN], label=str(N))     
    
    plt.title(params)
    plt.legend()
    plt.savefig(f'acc_plots/{nparam}.png')
    

    
    
    
    
    
    
    