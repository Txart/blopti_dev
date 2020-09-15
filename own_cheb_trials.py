# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:50:51 2020

@author: 03125327
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import math
import time

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

#%%
# Nonlinear heat equation Using Chebyshev:
#    du/dt = d/dx(A(u)*du/dx) + S, which is equivalent to solving
#    du/dt = A * u'' + dA/du * u'^2 + S
#    BC: u(-1) = 0; u'(1) = 0; IC: u(0) = 1
# Preparation for actual one

c_start_time = time.time()

N = 10
D,x = cheb(N)
x = x/5; D=D/5 # Dimension of the domain
D2 = D @ D # matrix multiplication

# IC
INI_VALUE = 100.0
v_ini = np.ones(shape=x.shape)*INI_VALUE
# BC
v_ini[-1] = 0

v_old = v_ini[:] # v_ini is used later for others

# source = P-ET
source = 2. # P- ET

def dif(u):
    # Dffusivity
    params = [2.0, 1.1, 1.0, 2.0, 2.3] # s0, s1, t0, t1, t2
    s0 = params[0]; s1 = params[1];
    t0 = params[1]; t1 = params[3]; t2 = params[4];
    
    D = t0/s1 * np.exp(t1 - t2*s0/s1) * np.power(u, t2/s1 - 1.)
    
    return D

def dif_u(u):
    # Derivative of diffusivity with respect to theta
    # Have to hardcode the derivative
    params = [2.0, 1.1, 1.0, 2.0, 2.3] # s0, s1, t0, t1, t2
    s0 = params[0]; s1 = params[1];
    t0 = params[1]; t1 = params[3]; t2 = params[4];
    D_prime = t0/s1**2 * (t2-s1) * np.exp(t1 - t2*s0/s1) * np.power(u, (t2 - 2*s1)/s1)
    
    return D_prime

def forward_Euler(v_old, dt):
    return v_old + dt*( dif_u(v_old) * (D @ v_old)**2 + dif(v_old) * D2 @ v_old + source)



def RK4(v_old, dt):
    # 4th order Runge-Kutta
    def rhs(u):
        # RHS of the PDE: du/dt = rhs(u)
        return dif_u(u) * (D @ u)**2 + dif(u) * D2 @ u + source
    # Diri BC have to be specified every time the rhs is evaluated!
    k1 = rhs(v_old); k1[-1] = 0 # BC
    k2 = rhs(v_old + dt/2*k1); k2[-1] = 0 # BC
    k3 = rhs(v_old + dt/2*k2); k3[-1] = 0 # BC
    k4 = rhs(v_old + dt*k3); k4[-1] = 0 # BC
    
    return v_old + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
    
 
# Solve iteratively
dt = 0.001
TIMESTEPS = 3
niter = int(TIMESTEPS/dt)
v_plot = [0]*(niter+1)
v_plot[0] = v_old
for i in range(niter):
    v_new = forward_Euler(v_old, dt)
    v_new[-1] = 0 # Diri BC
    nbc =  D[0,1:] @ v_new[1:] # Neumann BC
    v_new[0] = -1/D[0,0] * nbc
    v_old = v_new
     
    v_plot[i+1] = v_old

print(f"Cheb time(s) = {time.time() - c_start_time}")  

# Waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Chebyshev')

v_plot = np.array(v_plot)
v_plot = v_plot[0::int(niter/TIMESTEPS)] # NumPy slice -> start:stop:step

for j in range(v_plot.shape[0]):
    ys = j*np.ones(v_plot.shape[1])
    ax.plot(x,ys,v_plot[j,:])

  
#%%
# Check with fipy

import fipy as fp
from fipy.tools import numerix

N=10
# N = N+1 # cheby starts at pos=0

dx = 1.

f_start_time = time.time()

mesh = fp.Grid1D(nx=N, dx=dx)

v_fp = fp.CellVariable(name="v_fp", mesh=mesh, value=1.)

# BC
v_fp.constrain(0, where=mesh.facesLeft) # left BC is always Dirichlet
# v_fp.faceGrad.constrain(0. * mesh.faceNormals, where=mesh.facesRight) # right: flux=0


def dif_fp(u):
    params = [2.0, 1.1, 1.0, 2.0, 1.3] # s0, s1, t0, t1, t2
    s0 = params[0]; s1 = params[1];
    t0 = params[1]; t1 = params[3]; t2 = params[4];
    D = t0/s1 * numerix.exp(t1 - t2*s0/s1) * numerix.power(u, t2/s1 - 1.)
    
    return D

# Boussinesq eq. for theta
eq = fp.TransientTerm() == fp.ExplicitDiffusionTerm(coeff=dif_fp(v_fp)) + source



sol_fp = [0] * (niter+1) # returned quantity
sol_fp[0] = np.array(v_fp.value)

MAX_SWEEPS = 1

# for i in range(niter):
    
#     # Append to list
#     sol_fp[i] = v_fp.value
#     print(i, v_fp.value[0])
    
#     v_fp.updateOld()

#     res = 0.0
#     for r in range(MAX_SWEEPS):
#         # print(i, res)
#         resOld=res
#         res = eq.sweep(var=v_fp, dt=dt)
#         if abs(res - resOld) < 1e-3: break # it has reached to the solution of the linear system

# dt = 0.9*4/(2*D*N**2)
dt = 0.001

for i in range(niter):
    eq.solve(var=v_fp, dt=dt)
    sol_fp[i+1] = np.array(v_fp.value)

print(f"FiPy time(s) = {time.time() - f_start_time}")  

# Waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'FiPy, dx={dx}, N={N}')

plot_sol_fp = np.array(sol_fp)
plot_sol_fp = plot_sol_fp[0::int(niter/TIMESTEPS)] # NumPy slice -> start:stop:step
x_fp = np.linspace(-1, 1, num=N)

for j in range(plot_sol_fp.shape[0]):
    ys = j*np.ones(plot_sol_fp.shape[1])
    ax.plot(x_fp, ys, plot_sol_fp[j,:])

    
#%%
#    Compare fipy vs cheby 

# In order to compare, add 0 at beginning of fipy solution (which is in the centers of cells)
plot_sol_fp_padded = np.pad(array=plot_sol_fp, pad_width=[[0,0],[1,0]])
    
# Waterfall plot of difference
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('abs(FiPy-cheby)')

for j in range(plot_sol_fp.shape[0]):
    ys = j*np.ones(plot_sol_fp_padded.shape[1])
    ax.plot(x, ys, abs(plot_sol_fp_padded[j,:]-v_plot[j,:][::-1]))