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
    
    # Compute tolerance. Each day, a new tolerance because source changes
    _, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)
    rel_tol = rel_tolerance * np.linalg.norm(F)
    print(rel_tol)
    
    for i in range(0, MAX_INTERNAL_NITER):
        J, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)
        
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
    
    return v, time_spent

#%%
import fipy as fp
from fipy.tools import numerix

def solve_fipy_with_given_N(N, params):
    
    s1 = params[0]; s2 = params[1]
    t1 = params[2]; t2 = params[3]
    
    dx = 2.0/N
    dt = 1.0
    
    RELAXATION = 1/N

    f_start_time = time.time()
    
    mesh = fp.Grid1D(nx=N, dx=dx)
    
    v_fp = fp.CellVariable(name="v_fp", mesh=mesh, value=INI_VALUE, hasOld=True)
    
    # BC
    v_fp.constrain(0, where=mesh.facesLeft) # left BC is always Dirichlet
    # v_fp.faceGrad.constrain(0. * mesh.faceNormals, where=mesh.facesRight) # right: flux=0
    
    def dif_fp(u):
        b=-4.
        D = (numerix.exp(t1)/t2 * (numerix.power(s2 * numerix.exp(-s1) * u + numerix.exp(s2*b), t2/s2) - numerix.exp(t2*b))) / (s2 * u + numerix.exp(s1 + s2*b))
        
        return D

    # Boussinesq eq. for theta
    eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=dif_fp(v_fp)) + SOURCE
    
    MAX_SWEEPS = 10000

    v_fp.updateOld()

    res = 0.0
    for r in range(MAX_SWEEPS):
        # print(i, res)
        resOld=res
        res = eq.sweep(var=v_fp, dt=dt, underRelaxation=RELAXATION)
        if abs(res - resOld) < abs_tolerance: break # it has reached to the solution of the linear system

    
    time_spent = time.time() - f_start_time
    
    return  v_fp.value, time_spent

#%%
# Params
DIRI = 0
SOURCE = 2
INI_VALUE = 0.2
MAX_INTERNAL_NITER = 10000 # max niters to solve nonlinear algebraic eq of Newton's method

rel_tolerance = 1e-5
abs_tolerance = 1e-5


#%%
# Run accuracy tests
Ns = [10, 25, 50, 100, 200]

N_PARAMS = 10
rnd_params = np.random.rand(N_PARAMS,4) * 2

v_sols = [[] for i in range(N_PARAMS)]
times = [[] for i in range(N_PARAMS)]

v_sols_fipy = [[] for i in range(N_PARAMS)]
times_fipy = [[] for i in range(N_PARAMS)]

for nN, N in enumerate(Ns):
    for nparam, params in enumerate(rnd_params):
        v_sol, time_spent = solve_with_given_N(N, params)
        v_sol_fipy, time_spent_fipy = solve_fipy_with_given_N(N, params)
        
        v_sols[nparam].append(v_sol)
        v_sols_fipy[nparam].append(v_sol_fipy)
        times[nparam].append(time_spent)
        times_fipy[nparam].append(time_spent_fipy)
        
        

#%%
# Plot accuracies
        
cmap = plt.cm.Accent
cmaplist = [cmap(int(i)) for i in range(0, len(Ns))]

for nparam, params in enumerate(rnd_params):
    plt.figure(nparam, figsize=(8, 6), dpi=400)
    for nN, N in enumerate(Ns):
        x = np.linspace(0,2,N+1)
        x_fp = np.linspace(0,2,N)
        plt.plot(x, v_sols[nparam][nN], color=cmaplist[nN], label=str(N))
        plt.plot(x_fp, v_sols_fipy[nparam][nN], '--', color=cmaplist[nN], label=str(N) + ' fipy')
        
    
    plt.title(params)
    plt.legend()
    plt.savefig(f'acc_plots/{nparam}.png')
    
# Plot times
times_np = np.array(times)
times_fipy_np = np.array(times_fipy)
time_avgs = np.mean(times_np, axis=0)
time_avgs_fipy = np.mean(times_fipy_np, axis=0)

plt.figure('times')
plt.plot(Ns, time_avgs, 'o', label='implicit + fortran')
plt.plot(Ns, time_avgs_fipy, 'x', label='fipy')
plt.title('Comp times')
plt.legend()
plt.savefig('acc_plots/acc_comp_times.png')

    
    
    
    
    
    
    