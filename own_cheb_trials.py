# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:50:51 2020

@author: 03125327
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, lu_factor, lu_solve, solve_banded
import math
import time

#%%
plotOpt = True

#%%
"""
 Params 
"""

params = [ -0.9, 1, 1.6, 10]
s1 = params[0]; s2 = params[1]
t1 = params[2]; t2 = params[3]
SOURCE = -0.003 # P- ET
INI_VALUE = 0.5
DIRI_BC = 0.3
TIMESTEPS = 3

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

print('>>>>> Chebyshev started')
c_start_time = time.time()

N = 10
D,x = cheb(N)
x = x; D=D # Dimension of the domain
D2 = D @ D # matrix multiplication

# IC
v_ini = np.ones(shape=x.shape)*INI_VALUE
# BC
v_ini[-1] = 0

v_old = v_ini[:] # v_ini is used later for others



def S(u, b):
    return s2 * u + np.exp(s1 + s2*b)

def T(u, b):
    return np.exp(t1)/t2 * (np.power(s2 * np.exp(-s1) * u + np.exp(s2*b), t2/s2) - np.exp(t2*b))

def dif(u):
    # Diffusivity
    b=-4.
    return T(u, b) * np.power(S(u, b), -1)

    
def dif_u(u):
    b = -4.
    # Derivative of diffusivity with respect to theta
    # Have to hardcode the derivative
    T_prime = np.exp(t1-s1) * np.power(s2/np.exp(s1)* u + np.exp(s2*b), (t2-s2)/s2)
    # S_prime = s2
    

    diffusivity_prime = T_prime/S(u,b) - T(u, b) * s2 * np.power(S(u, b), -2)        
    
    return diffusivity_prime

def dif_simple(u):
    return np.ones(u.shape)

def dif_u_simple(u):
    return 0.

def forward_Euler(v_old, dt):
    return v_old + dt*( dif_u_simple(v_old) * (D @ v_old)**2 + dif_simple(v_old) * D2 @ v_old + SOURCE)



def RK4(v_old, dt):
    # 4th order Runge-Kutta
    def rhs(u):
        # RHS of the PDE: du/dt = rhs(u)
        return dif_u(u) * (D @ u)**2 + dif(u) * D2 @ u + SOURCE
    # Diri BC have to be specified every time the rhs is evaluated!
    k1 = rhs(v_old); k1[-1] = 0 # BC
    k2 = rhs(v_old + dt/2*k1); k2[-1] = 0 # BC
    k3 = rhs(v_old + dt/2*k2); k3[-1] = 0 # BC
    k4 = rhs(v_old + dt*k3); k4[-1] = 0 # BC
    
    return v_old + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
    
 
# Solve iteratively
dt = 0.0001

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

if False:
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
import copy

N = 10 # cheby starts at pos=0

dx = 2/N

print('\n >>>>> FiPy started')

f_start_time = time.time()

mesh = fp.Grid1D(nx=N, dx=dx)

v_fp = fp.CellVariable(name="v_fp", mesh=mesh, value=INI_VALUE, hasOld=True)

# BC
v_fp.constrain(DIRI_BC, where=mesh.facesLeft) # left BC is always Dirichlet
# v_fp.faceGrad.constrain(0. * mesh.faceNormals, where=mesh.facesRight) # right: flux=0

b = -4.

def dif_fp(u):
    b=-4.
    D = (numerix.exp(t1)/t2 * (numerix.power(s2 * numerix.exp(-s1) * u + numerix.exp(s2*b), t2/s2) - numerix.exp(t2*b))) / (s2 * u + numerix.exp(s1 + s2*b))
    
    return D

def dif_fp_simple(u):
    # Simpler diffusivity, for checks
    return 1.

# Boussinesq eq. for theta
eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=(numerix.exp(t1)/t2 * (numerix.power(s2 * numerix.exp(-s1) * v_fp + numerix.exp(s2*b), t2/s2) - numerix.exp(t2*b))) / (s2 * v_fp + numerix.exp(s1 + s2*b))) + SOURCE


dt = 1.0
niter = int(TIMESTEPS/dt)

sol_fp = [0] * (TIMESTEPS+1) # returned quantity
sol_fp[0] = np.array(v_fp.value)

MAX_SWEEPS = 10000

for i in range(TIMESTEPS):
    
    v_fp.updateOld()

    res = 0.0
    for r in range(MAX_SWEEPS):
        resOld=res
        res = eq.sweep(var=v_fp, dt=dt, underRelaxation=1.0)
        if abs(res - resOld) < 1e-5: break # it has reached to the solution of the linear system
    
    print(r, res)
      # Append to list
    sol_fp[i+1] = copy.copy(v_fp.value[:])
    print(i, v_fp.value)


# for i in range(niter):
#     eq.solve(var=v_fp, dt=dt)
#     sol_fp[i+1] = np.array(v_fp.value)

print(f"FiPy time(s) = {time.time() - f_start_time}")  

if plotOpt:
    # Waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'FiPy, dx={dx}, N={N}')
    
    plot_sol_fp = np.array(sol_fp)
    # plot_sol_fp = plot_sol_fp[0::int(niter/TIMESTEPS)] # NumPy slice -> start:stop:step
    x_fp = np.linspace(0, 2, num=N)
    
    for j in range(plot_sol_fp.shape[0]):
        ys = j*np.ones(plot_sol_fp.shape[1])
        ax.plot(x_fp, ys, plot_sol_fp[j,:])

    
#%%
# Compare fipy vs cheby 

if plotOpt:
    # In order to compare, interpolate the chebyshev and evaluate
    # at fipy mesh centers: x = (0.5, 1.5 , ...)
    import scipy.interpolate.interpolate as interp
    
    cheby_interp = interp.interp1d(x + 1, v_old) # The +1 is to begin at x=0
    x_fp = mesh.cellCenters.value[0]
    cheby_interpolated = cheby_interp(x_fp[0:N-1])
    
    # Plot together
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_fp, plot_sol_fp[-1], label='fipy')
    ax.plot(x+1, v_old, label='chebyshev')
    plt.legend()
    
    # Plot of difference
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('abs(FiPy-cheby)')
    
    ax.plot(x_fp[0:N-1], abs(plot_sol_fp[-1][0:N-1] - cheby_interpolated))
    
    
#%%
    # Simple Cheby: d/dx(A(u) du/dx) -> D(A(u_cheby)*Du_cheby)
    # Direct computation of the 2nd derivative

# print('\n >>>>> Simple chebyshev started')

# c_start_time = time.time()

# N = 10
# D,x = cheb(N)
# x = x; D=D # Dimension of the domain
# # D2 = D @ D # matrix multiplication

# # IC
# # We have to interpolate the initial value to the cheby points x!
# # In this constant case, it's the same
# v_ini = np.ones(shape=x.shape)*INI_VALUE
# # BC
# v_ini[-1] = 0

# v_old = v_ini[:] # v_ini is used later for others




# def forward_Euler(v_old, dt):
#     return v_old + dt*( D @ (dif_simple(v_old) * (D @ v_old)) + SOURCE)

    
 
# # Solve iteratively
# dt = 0.00001
# TIMESTEPS = 3
# niter = int(TIMESTEPS/dt)
# v_plot = [0]*(niter+1)
# v_plot[0] = v_old
# for i in range(niter):
#     v_new = forward_Euler(v_old, dt)
#     v_new[-1] = 0 # Diri BC
#     nbc =  D[0,1:] @ v_new[1:] # Neumann BC
#     v_new[0] = -1/D[0,0] * nbc
#     v_old = v_new
    
#     v_plot[i+1] = v_old

# print(f"Cheb Simple time(s) = {time.time() - c_start_time}")  

# if plotOpt:
#     # Waterfall plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Chebyshev New')
    
#     v_plot = np.array(v_plot)
#     v_plot = v_plot[0::int(niter/TIMESTEPS)] # NumPy slice -> start:stop:step
    
#     for j in range(v_plot.shape[0]):
#         ys = j*np.ones(v_plot.shape[1])
#         ax.plot(x,ys,v_plot[j,:])



#%%
# # Cheby implicit
    
# print('\n >>>>> Chebyshev implicit started')

# c_start_time = time.time()

# N = 10

# DELTA_t = 1.0 # in days
# v_ini = np.ones(shape=x.shape)*INI_VALUE
# # BC
# v_ini[-1] = 0

# v_old = v_ini[:] # v_ini is used later for others

# def dif_u_u_simple(u):
#     return 0.

# # Picard iteration
# def picard(v, source, DELTA_t):
#     a = DELTA_t * ((dif_u_simple(v)*(D @ v)) @ D - np.eye(N+1) + dif_simple(v)*np.eye(N+1) @ D2)
#     b = v + source
#     return np.linalg.solve(a,b)

# def newton(v, v0, source, DELTA_t):
#     # v0 is the value in the previous iteration
#     a = DELTA_t*((2*dif_u_simple(v)*(D@v)*np.eye(N+1)) @ D + dif_u_u_simple(v)*(D@v)*(D@v)*np.eye(N+1) +
#           (dif_simple(v)*np.eye(N+1))@D2 + dif_u_simple(v)*(D2@v)*np.eye(N+1)) - np.eye(N+1)
#     b = DELTA_t*(dif_u_simple(v)*(D@v)*(D@v) + dif_simple(v)*(D2@v) + source) + v0 - v
    
#     return np.linalg.solve(a,b)    
    

# TIMESTEPS = 3
# weight = 0.0001
# internal_niter = 10000
# v_plot = [0]*(TIMESTEPS+1)
# v_plot[0] = v_old[:]
# for t in range(TIMESTEPS):
#     v0 = v_old[:]
#     for i in range(internal_niter):
#         v_new = v_old + weight * newton(v_old, v0, SOURCE, DELTA_t)
#         v_new[-1] = 0 # Diri BC
#         nbc =  D[0,1:] @ v_new[1:] # Neumann BC
#         v_new[0] = -1/D[0,0] * nbc
#         v_old = v_new[:] 
#     v_plot[t+1] = v_old[:]

# print(f"Cheb implicit time(s) = {time.time() - c_start_time}")  

# if plotOpt:
#     # Waterfall plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Chebyshev implicit')
    
#     v_plot = np.array(v_plot)
    
    
#     for j in range(v_plot.shape[0]):
#         ys = j*np.ones(v_plot.shape[1])
#         ax.plot(x,ys,v_plot[j,:])

#%%
# # Cheby simplewith implicit backward Euler Neton-Rhapson method

# print('\n >>>>> Chebyshev implicit started')

# c_start_time = time.time()

# N = 10

# DELTA_t = 1.0 # in days
# v_ini = np.ones(shape=x.shape)*INI_VALUE
# # BC
# v_ini[-1] = 0

# v_old = v_ini[:] # v_ini is used later for others

# def newton(v, v0, source, DELTA_t):
#     # v0 is the value in the previous iteration
#     a = DELTA_t*((D @ (dif_u_simple(v) * (D @ v)))*np.eye(N+1) + (D @ dif_simple(v)) * D) - np.eye(N+1)
#     b = DELTA_t*(D @ (dif_simple(v)*(D @ v)) + source) + v0 - v
    
#     return np.linalg.solve(a,b)    
    

# TIMESTEPS = 3
# weight = 0.0001
# internal_niter = 10000
# v_plot = [0]*(TIMESTEPS+1)
# v_plot[0] = v_old[:]
# for t in range(TIMESTEPS):
#     v0 = v_old[:]
#     for i in range(internal_niter):
#         v_new = v_old + weight * newton(v_old, v0, SOURCE, DELTA_t)
#         v_new[-1] = 0 # Diri BC
#         nbc =  D[0,1:] @ v_new[1:] # Neumann BC
#         v_new[0] = -1/D[0,0] * nbc
#         v_old = v_new[:] 
#     v_plot[t+1] = v_old[:]

# print(f"Cheb implicit time(s) = {time.time() - c_start_time}")  

# if plotOpt:
#     # Waterfall plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Chebyshev simple implicit')
    
#     v_plot = np.array(v_plot)
    
    
#     for j in range(v_plot.shape[0]):
#         ys = j*np.ones(v_plot.shape[1])
#         ax.plot(x,ys,v_plot[j,:])


#%%
# Implicit Backwards Euler, Newton method with finite differences in space and time
    
    #TODO: Measure time with different constructions of J
    # Implement scipy tridiagonal matrix solution

def to_banded_matrix(a, lower=1, upper=1):
    """
    a is a numpy square matrix
    this function converts a square matrix to diagonal ordered form
    returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
    """
    n = a.shape[1]
    assert(np.all(a.shape ==(n,n)))
    
    ab = np.zeros((2*n-1, n))
    
    for i in range(n):
        ab[i,(n-1)-i:] = np.diagonal(a,(n-1)-i)
        
    for i in range(n-1): 
        ab[(2*n-2)-i,:i+1] = np.diagonal(a,i-(n-1))

    mid_row_inx = int(ab.shape[0]/2)
    upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
    keep_rows = upper_rows+lower_rows
    ab = ab[keep_rows,:]


    return ab


"""
  Construct Jacobian matrix and F to solve newton method:
      J x = -F
""" 
def jacobian(v, delta_t, N):
    J = np.zeros(shape=(N+1,N+1)) # Dirichlet BC in the matrix
    
    for (i,j), _ in np.ndenumerate(J):
    
        if i>=1 and i<N:
        
            J[i,i-1] = e * (-a_u(v[i-1])*v[i-1] - a(v[i]) - a(v[i-1]) + a_u(v[i-1])*v[i])
            
            J[i,i] = e*(-a_u(v[i])*v[i-1] + 2*a_u(v[i])*v[i] - a_u(v[i])*v[i+1] + a(v[i+1]) +
                        2*a(v[i]) + a(v[i-1])) + 1/delta_t
            
            J[i,i+1] = e*(a_u(v[i+1])*v[i] - a(v[i+1]) - a(v[i]) - a_u(v[i+1])*v[i+1])

    # BC
    # Diri in x=0
    J[0,0] = 1
    # Neumann with diffusivity(u(L))*u'(L)=C in x=N
    aL = a(2*dx*C/a(v[N]) + v[N-1])
    J[N,N] = e*(-a_u(v[N-1])*v[N-1] + 2*a_u(v[N])*v[N] + aL + 2*a(v[N]) + a(v[N-1])) + 1/dt - C * a_u(v[N])/(a(v[N]))**2
    J[N,N-1] = e*(-a_u(v[N-1])*v[N-1] + a_u(v[N-1])*v[N] - aL - 2*a(v[N]) - a(v[N-1])) -dx*e*(a_u(2*dx*C/a(v[N]) + v[N-1]))
    
    return J

def jacobian_and_F_vectorial(v, v_old, delta_t, N, D, D_u, diri_bc, source):
    """
    Vectorial construction of Jacobian matrix for newton-rhapson method.
    By construction, C=0. So no-flux BC in the upper part of the canal.

    Parameters
    ----------
    v : np.array 1d
        solution vector
    v_old : np.array 1d
        solution vector at previous timestep
    delta_t : float
        dt in days
    N : float
        dimensions.
    D : function
        Diffusivity
    D_u : function
        Derivative of the diffusivity 
    diri_bc : float
        Dirichlet bc value. Imposed at 0th position
    source : float
        Source term in PDE

    Returns
    -------
    J : 2d np.array
        jacobian matrix

    """
    # def of vectors
    # "surname" u means up, surname d means down
    
    vu = np.append(v, 0)[1:]
    vd = np.insert(v, 0, 0)[:-1]
    a = D(v)
    au = D(vu)
    ad = D(vd)
    a_prime = D_u(v)
    a_primeu = D_u(vu)
    a_primed = D_u(vd)
    
    # Jacobian
    # compute vectors that will go in matrix
    j_diag = e*(-a_prime*(vd + 2*v - vu) + au + 2*a + ad) + 1/delta_t
    j_superdiag = e*(a_primeu*(v-vu) - au - a)
    j_subdiag = e * (a_primed*(v-vd) - a - ad)
    
    # put vectors into matrix positions
    # np.diag does a nice job here!
    J = np.diag(j_diag) + np.diag(j_superdiag[:-1], 1) + np.diag(j_subdiag[1:], -1)
    
    J_banded = np.zeros((3,N+1))
    J_banded[1] = j_diag
    J_banded[0][1:] = j_superdiag[:-1]; J_banded[0,1] = 0
    J_banded[2][:-1] = j_subdiag[1:]
    
    # BC
    J[0] = np.zeros(N+1)
    J[0,0] = 1
    aL = D(v[N-1])
    J[N] = np.zeros(N+1)
    J[N,N] = e*(-D_u(v[N-1])*v[N-1] + 2*D_u(v[N])*v[N] + aL + 2*D(v[N]) + D(v[N-1])) + 1/dt
    J[N,N-1] = e*(-D_u(v[N-1])*v[N-1] + D_u(v[N-1])*v[N] - aL - 2*D(v[N]) - D(v[N-1])) -dx*e*(D_u(v[N-1]))
    
    # Diri
    J_banded[1,0] = 1
    # Neumann
    aL = D(v[N-1])
    J_banded[1,N] = J[N,N] = e*(-D_u(v[N-1])*v[N-1] + 2*D_u(v[N])*v[N] + aL + 2*D(v[N]) + D(v[N-1])) + 1/dt
    J_banded[2,N-1] = e*(-D_u(v[N-1])*v[N-1] + D_u(v[N-1])*v[N] - aL - 2*D(v[N]) - D(v[N-1])) -dx*e*(D_u(v[N-1]))
    
    # F
    F = -e*((a + ad)*vd - v*(au + 2*a + ad) + vu*(au + a)) - source - v_old/delta_t + v/delta_t
    # BC
    F[0] = diri_bc # Dirichlet in x=0
    # Neumann with flux u'(x=N) = C/diffusivity'(u(x=N))
    aN = D(v[N-1])
    F[N] = -e*((D(v[N]) + D(v[N-1]))*v[N-1] -v[N]*(aN + 2*D(v[N]) + D(v[N-1])) +
                        v[N-1]*(aN + D(v[N]))) - source - v_old[N]/delta_t + v[N]/delta_t
    
    
    return J, J_banded, F


def jacobian_diag_ordered(v, delta_t, N):
    """ This is not right. I give up!"""
    
    J_banded = np.zeros(shape=(3,N+1))
    
    for i in range(0,N):
        for j in range(0,N):
            if i==j: # diag elements of J
                J_banded[1,j] = e*(-a_u(v[i])*v[i-1] + 2*a_u(v[i])*v[i] - a_u(v[i])*v[i+1] + a(v[i+1]) +
                            2*a(v[i]) + a(v[i-1])) + 1/delta_t
                
            elif i==j+1: # lower diagonal elements of J
                J_banded[2,j] = e*(a_u(v[i+1])*v[i] - a(v[i+1]) - a(v[i]) - a_u(v[i+1])*v[i+1])
                
            elif i==j-1: # upper diag of J
                J_banded[0,j] = e*(-a_u(v[i])*v[i-1] + 2*a_u(v[i])*v[i] - a_u(v[i])*v[i+1] + a(v[i+1]) +
                        2*a(v[i]) + a(v[i-1])) + 1/delta_t
    
    # Diri in x=0
    J_banded[1,0] = 1
    # Neumann with diffusivity(u(L))*u'(L)=C in x=N
    aL = a(2*dx*C/a(v[N]) + v[N-1])
    J_banded[1,N] = e*(-a_u(v[N-1])*v[N-1] + 2*a_u(v[N])*v[N] + aL + 2*a(v[N]) + a(v[N-1])) + 1/dt - C * a_u(v[N])/(a(v[N]))**2
    J_banded[2,N-1] = e*(-a_u(v[N-1])*v[N-1] + a_u(v[N-1])*v[N] - aL - 2*a(v[N]) - a(v[N-1])) -dx*e*(a_u(2*dx*C/a(v[N]) + v[N-1]))
    
    return J_banded
    
def F_newton(v, source, delta_t, v_old, diri_bc):
    # Returns right hand side of Newton method: Jx = -F
    # v_old is solution variable in previous timestep
    F = np.zeros(N+1)
    for i,_ in enumerate(F):
        if i>=1 and i<N:
            F[i] = -e*((a(v[i]) + a(v[i-1]))*v[i-1] -v[i]*(a(v[i+1]) + 2*a(v[i]) + a(v[i-1])) +
                        v[i+1]*(a(v[i+1]) + a(v[i])))
    
    F = F - source - v_old/delta_t + v/delta_t
    
    # BC
    F[0] = diri_bc # Dirichlet in x=0
    # Neumann with flux u'(x=N) = C/diffusivity'(u(x=N))
    uN = 2*dx*C/a(v[N]) + v[N-1]
    aN = a(uN)
    F[N] = -e*((a(v[N]) + a(v[N-1]))*v[N-1] -v[N]*(aN + 2*a(v[N]) + a(v[N-1])) +
                        uN*(aN + a(v[N]))) - source - v_old[N]/delta_t + v[N]/delta_t
    
    return F

print('\n >>>>> Finite diff Python started')

c_start_time = time.time()


rel_tolerance = 1e-5
abs_tolerance = 1e-5

N = 10
dt = 1.0 # in days
dx = 2/N # in m 
v_ini = np.ones(shape=N+1)*INI_VALUE

# BC
DIRI = 0.
C = 0. # Neumann BC u'(x=N) = C/diffusivity'(u(x=N)). C=0. corresponds to no-flux
v_ini[0] = DIRI

v = v_ini[:]
v_old = v_ini[:] # in the previous timestep




# Relaxation parameter
weight = 0.1

# Notation
a = dif
a_u = dif_u
e = 1/(2*dx**2)

J = jacobian(v, dt, N)
J, J_banded, F = jacobian_and_F_vectorial(v, v_old, dt, N, dif, dif_u, DIRI, SOURCE)

# J_banded = jacobian_diag_ordered(v, dt, N)
# F = F_newton(v, SOURCE, dt, v_old, DIRI)

# vectorial version
# J_v = jacobian_vectorial(v, dt, N, a, a_u)

# Plotting stuff
v_plot = [0]*(TIMESTEPS+1)
v_plot[0] = v_ini[:]


MAX_INTERNAL_NITER = 10000 # max niters to solve nonlinear algebraic eq of Newton's method

for t in range(TIMESTEPS):
    # Update source
    source = SOURCE
    
    # Update BC
    DIRI = DIRI
    # No-flux in the right all the time
    
    # Compute tolerance. Each day, a new tolerance because source changes
    rel_tol = rel_tolerance * np.linalg.norm(F_newton(v, SOURCE, dt, v_old, DIRI))

    for i in range(0, MAX_INTERNAL_NITER):
        # J = jacobian(v, dt, N)
        # J, J_banded, F = jacobian_and_F_vectorial(v, v_old, dt, N, dif_simple, dif_u_simple, DIRI, SOURCE)
        J, _, F = jacobian_and_F_vectorial(v, v_old, dt, N, a, a_u, DIRI, SOURCE)
        # F = F_newton(v, SOURCE, dt, v_old, DIRI)
        
        eps_x = np.linalg.solve(J,-F)
        # eps_x = solve_banded((1,1), J_banded, -F, overwrite_ab=True, overwrite_b=True, check_finite=False)
        # eps_x = lu_solve(lu_factor(J), -F)
        # eps_x = solve(J, -F)
        v = v + weight*eps_x

        # stopping criterion
        residue = np.linalg.norm(F) - rel_tol
        if residue < abs_tolerance:
            print(f'Solution of the Newton linear system in {i} iterations')
            break
    
    v_old = v[:]
    v_plot[t+1] = v[:]
    print(i, v_old)
    
    

print(f"Finite diff implicit time(s) = {time.time() - c_start_time}") 

if plotOpt:
    # Waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Finite diff implicit')
    
    v_plot = np.array(v_plot)
    x = np.linspace(0,dx*N, N+1)
    
    for j in range(v_plot.shape[0]):
        ys = j*np.ones(v_plot.shape[1])
        ax.plot(x,ys,v_plot[j,:])

#%%
# Compare fipy vs finite diff implicit 
if plotOpt:
    # In order to compare, interpolate the finite diff and evaluate
    # at fipy mesh centers: x = (0.5, 1.5 , ...)
    import scipy.interpolate.interpolate as interp
    
    fdiff_interp = interp.interp1d(x, v_old)
    x_fp = mesh.cellCenters.value[0][0:-1]
    fdiff_interpolated = fdiff_interp(x_fp)
    
    # Plot together
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_fp, plot_sol_fp[-1][:-1], label='fipy')
    ax.plot(x, v_old, label='finite diff implicit')
    plt.legend()
    
    # Plot of difference
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('abs(FiPy-finite diff implicit)')
    
    ax.plot(x_fp, abs(plot_sol_fp[-1][:-1] - fdiff_interpolated))


#%% FORTRAN BUSINESS
    
import fd # own fortran functions

# TODO: USE np.asfortranarray()  before calling to fortran FUNCTIONS IN THE FUTURE

print('\n >>>>> Finite diff FORTRAN started')
c_start_time = time.time()


rel_tolerance = 1e-5
abs_tolerance = 1e-5

N = 10

dx = 0.2 # in m 

v_ini = np.ones(shape=N+1)*INI_VALUE

C = 0. # Neumann BC u'(x=N) = C/diffusivity'(u(x=N)). C=0. corresponds to no-flux
v_ini[0] = DIRI_BC

v = v_ini[:]
v_old = v_ini[:] # in the previous timestep

b = np.ones(shape=v.shape) * (-4)

# Relaxation parameter
weight = 0.01

# J, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)

# Plotting stuff
v_plot = [0]*(TIMESTEPS+1)
v_plot[0] = v_ini[:]


MAX_INTERNAL_NITER = 100000 # max niters to solve nonlinear algebraic eq of Newton's method

for t in range(TIMESTEPS):
    # Update source

    
    # Update BC
    v[0] = DIRI_BC
    v_old[0] = DIRI_BC
    # No-flux in the right all the time
    
    # Compute tolerance. Each day, a new tolerance because source changes
    _, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI_BC, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)
    rel_tol = rel_tolerance * np.linalg.norm(F)
    print(rel_tol)
    
    for i in range(0, MAX_INTERNAL_NITER):
        J, F = fd.j_and_f(n=N, v=v, v_old=v_old, b=b, delta_t=dt, delta_x=dx, diri_bc=DIRI_BC, s1=s1, s2=s2, t1=t1, t2=t2, source=SOURCE)
        
        eps_x = np.linalg.solve(J,-F)
        # eps_x = solve_banded((1,1), J_banded, -F, overwrite_ab=True, overwrite_b=True, check_finite=False)
        # eps_x = lu_solve(lu_factor(J), -F)
        # eps_x = solve(J, -F)
        v = v + weight*eps_x
        # v = np.where(v<0, 0, v)

        # stopping criterion
        residue = np.linalg.norm(F[1:]) - rel_tol
        if residue < abs_tolerance:
            print(f'Solution of the Newton linear system in {i} iterations')
            break
    
    # Early stopping criterion: thta cannot be negative
    if np.any(v < 0) or np.any(np.isnan(v)):
        print(v)
        raise ValueError('NEGATIVE V FOUND, ABORTING')
    
    v_old = v[:]
    v_plot[t+1] = v[:]
    print(i, v_old)
    
print(f"Finite diff with fortran-constructed J and F (s) = {time.time() - c_start_time}")
    
#%%
# Compare fipy vs finite diff implicit 
if plotOpt:
    # In order to compare, interpolate the finite diff and evaluate
    # at fipy mesh centers: x = (0.5, 1.5 , ...)
    import scipy.interpolate.interpolate as interp
    
    fdiff_interp = interp.interp1d(x, v_old)
    x_fp = mesh.cellCenters.value[0][0:-1]
    fdiff_interpolated = fdiff_interp(x_fp)
    
    # Plot together
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_fp, plot_sol_fp[-1][:-1], label='fipy')
    ax.plot(x, v_old, label='half fortran')
    plt.legend()
    
    # Plot of difference
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('abs(FiPy-halfFortran)')
    
    ax.plot(x_fp, abs(plot_sol_fp[-1][:-1] - fdiff_interpolated))
    


if plotOpt:
    # Waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Finite diff Fortran')
    
    v_plot = np.array(v_plot)
    x = np.linspace(0,dx*N, N+1)
    
    for j in range(v_plot.shape[0]):
        ys = j*np.ones(v_plot.shape[1])
        ax.plot(x,ys,v_plot[j,:])

#%%
# Compare fipy vs finite diff implicit  
# if plotOpt:
#     # In order to compare, interpolate the finite diff and evaluate
#     # at fipy mesh centers: x = (0.5, 1.5 , ...)
#     import scipy.interpolate.interpolate as interp
    
#     fdiff_interp = interp.interp1d(x, v_old)
#     x_fp = mesh.cellCenters.value[0][0:-1]
#     fdiff_interpolated = fdiff_interp(x_fp)
    
#     # Plot together
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(x_fp, plot_sol_fp[-1][:-1], label='fipy')
#     ax.plot(x, v_old, label='finite diff Fortran')
#     plt.legend()
    
#     # Plot of difference
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title('abs(FiPy-finite diff implicit)')
    
#     ax.plot(x_fp, abs(plot_sol_fp[-1][:-1] - fdiff_interpolated))
    
    
    
    
# #%% FORTRAN BUSINESS ALL IN
# raise NotImplementedError
# import fdallin # own fortran functions

# # TODO: USE np.asfortranarray()  before calling to fortran FUNCTIONS IN THE FUTURE

# c_start_time = time.time()

# print('\n >>>>> FORTRAN ALL IN started')

# rel_tolerance = 1e-5
# abs_tolerance = 1e-5

# N = 10
# dt = 1.0 # in days
# dx = 0.2 # in m 
# v_ini = np.ones(shape=N+1)*INI_VALUE

# # BC
# DIRI = 0.
# C = 0. # Neumann BC u'(x=N) = C/diffusivity'(u(x=N)). C=0. corresponds to no-flux
# v_ini[0] = DIRI

# v = v_ini[:]
# v_old = v_ini[:] # in the previous timestep




# # Relaxation parameter
# weight = 0.1

# # Notation
# a = dif_simple
# a_u = dif_u_simple
# e = 1/(2*dx**2)


# # Plotting stuff
# v_plot = [0]*(TIMESTEPS+1)
# v_plot[0] = v_ini[:]


# MAX_INTERNAL_NITER = 10000 # max niters to solve nonlinear algebraic eq of Newton's method

# for t in range(TIMESTEPS):
#     # Update source
#     source = SOURCE
    
#     # Update BC
#     DIRI = DIRI
#     # No-flux in the right all the time
    
#     # Compute tolerance. Each day, a new tolerance because source changes
#     _, _, F = jacobian_and_F_vectorial(v, v_old, dt, N, a, a_u, DIRI, SOURCE) 
#     rel_tol = rel_tolerance * np.linalg.norm(F)

#     print('code is working up to here')
#     # call to fortran function using lapack and everything.
#     v = fdallin.fin_diff.finite_diff(v=v, v_old=v_old, b=b, n=N, dt=dt, dx=dx, source=source, s1=s1, s2=s2, t1=t1, t2=t2,
#                         diri_bc=DIRI, rel_tol=rel_tol, abs_tolerance=abs_tolerance,
#                         weight=weight, max_internal_niter=MAX_INTERNAL_NITER)
    
#     print(f'\n also here, this is v: {v}')
#     v_old = v[:]
#     v_plot[t+1] = v[:]
    
    

# print(f"Finite diff with fortran-constructed J and F (s) = {time.time() - c_start_time}") 

# if plotOpt:
#     # Waterfall plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Finite diff Fortran')
    
#     v_plot = np.array(v_plot)
#     x = np.linspace(0,dx*N, N+1)
    
#     for j in range(v_plot.shape[0]):
#         ys = j*np.ones(v_plot.shape[1])
#         ax.plot(x,ys,v_plot[j,:])

# #%%
# # Compare fipy vs finite diff implicit  
# if plotOpt:
#     # In order to compare, interpolate the finite diff and evaluate
#     # at fipy mesh centers: x = (0.5, 1.5 , ...)
#     import scipy.interpolate.interpolate as interp
    
#     fdiff_interp = interp.interp1d(x, v_old)
#     x_fp = mesh.cellCenters.value[0][0:-1]
#     fdiff_interpolated = fdiff_interp(x_fp)
    
#     # Plot together
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(x_fp, plot_sol_fp[-1][:-1], label='fipy')
#     ax.plot(x, v_old, label='finite diff Fortran')
#     plt.legend()
    
#     # Plot of difference
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title('abs(FiPy-finite diff implicit)')
    
#     ax.plot(x_fp, abs(plot_sol_fp[-1][:-1] - fdiff_interpolated))

#%%
# Plot S, T, dif and dif prime        
from matplotlib.widgets import Slider
        
def S(u, b, s1, s2):
    return s2 * (u + np.exp(s1 + s2*b)/s2)

def T(u, b, s1, s2, t1, t2):
    return np.exp(t1)/t2 * (np.power(s2 * np.exp(-s1) * u + np.exp(s2*b), t2/s2) - np.exp(t2*b))

def S_of_zeta(zeta, b, s1, s2):
    return np.exp(s1 + s2*zeta)

def T_of_zeta(zeta, b, t1, t2):
    return np.exp(t1)/t2 * (np.exp(t2*zeta) - np.exp(t2*b))

def dif_of_zeta(zeta, b, s1, s2, t1, t2):
    return T_of_zeta(zeta, b, t1, t2)/S_of_zeta(zeta, b, s1, s2)

def dif(u, b, s1, s2, t1, t2):
    # Diffusivity
    return T(u, b, s1, s2, t1, t2) * np.power(S(u, b, s1, s2), -1)

def dif_prime(u, b, s1, s2, t1, t2):
    # Derivative of diffusivity with respect to theta
    # Have to hardcode the derivative
    T_prime = np.exp(t1-s1) * np.power(s2/np.exp(s1)* u + np.exp(s2*b), (t2-s2)/s2)
    # S_prime = s2

    diffusivity_prime = (S(u, b, s1, s2) * T_prime - 
                         T(u, b, s1, s2, t1, t2) * s2) * np.power(S(u, b, s1, s2), -2)        
    
    return diffusivity_prime

def zeta_from_theta(x, b, s1, s2):
    return np.log(np.exp(s2*b) + s2*np.exp(-s1)*x) / s2

def theta_from_zeta(zeta, b, s1, s2):
    return np.exp(s1)/s2 * (np.exp(s2*zeta) - np.exp(s2*b))



theta = np.linspace(0,5,100)
zeta = zeta_from_theta(theta, -4, s1, s2)

inSpyder = True # interactive plots in the iPython console
if inSpyder and plotOpt:
    # theta interactive plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.75)
    # sliders
    ax_s1 = fig.add_axes([0.3, 0.73, 0.4, 0.05])
    ax_s2 = fig.add_axes([0.3, 0.80, 0.4, 0.05])
    ax_t1 = fig.add_axes([0.3, 0.87, 0.4, 0.05])
    ax_t2 = fig.add_axes([0.3, 0.94, 0.4, 0.05])
    
    # Create sliders
    s_s1 = Slider(ax=ax_s1, label='s1 ', valmin=-5, valmax=100.0,
                  valfmt=' %1.1f', facecolor='#cc7000')
    s_s2 = Slider(ax=ax_s2, label='s2 ', valmin=-5, valmax=100, 
                 valinit=1, valfmt='%1.1f ', facecolor='#cc7000')
    s_t1 = Slider(ax=ax_t1, label='t1 ', valmin=-5, valmax=100.0,
                  valfmt=' %1.1f', facecolor='#cc7000')
    s_t2 = Slider(ax=ax_t2, label='t2 ', valmin=-5, valmax=100, 
                 valinit=1, valfmt='%1.1f ', facecolor='#cc7000')
    
    f_S, = ax.plot(theta, S(theta, -4, s1, s2), label='S')
    f_z, = ax.plot(theta, zeta, label='zeta')
    f_T, = ax.plot(theta, T(theta, -4, s1, s2, t1, t2), label='T')
    f_d, = ax.plot(theta, dif(theta, -4, s1, s2, t1, t2), label='dif')
    f_dp, = ax.plot(theta, dif_prime(theta, -4, s1, s2, t1, t2), label='dif_prime')
    ax.set_title('function vs theta')
    ax.set_xlabel('theta')
    ax.legend()
    
    def update_theta(val):
        s1 = s_s1.val
        s2 = s_s2.val
        t1 = s_t1.val
        t2 = s_t2.val
        
        f_S.set_data(theta, S(theta, -4, s1, s2))
        f_z.set_data(theta, zeta_from_theta(theta, -4, s1, s2))
        f_T.set_data(theta, T(theta, -4, s1, s2, t1, t2))
        f_d.set_data(theta, dif(theta, -4, s1, s2, t1, t2))
        f_dp.set_data(theta, dif_prime(theta, -4, s1, s2, t1, t2))
    
    s_s1.on_changed(update_theta)
    s_s2.on_changed(update_theta)
    s_t1.on_changed(update_theta)
    s_t2.on_changed(update_theta)

    # zeta interactive plots\
    zeta = np.linspace(-4, 1, num=100)
    theta = theta_from_zeta(zeta, -4, s1, s2)
    figz = plt.figure()
    axz = figz.add_subplot(111)
    axz.set_xlim(-4,100)
    figz.subplots_adjust(bottom=0.2, top=0.68)
    # sliders
    axz_s1 = figz.add_axes([0.3, 0.94, 0.4, 0.05])
    axz_s2 = figz.add_axes([0.3, 0.87, 0.4, 0.05])
    axz_t1 = figz.add_axes([0.3, 0.80, 0.4, 0.05])
    axz_t2 = figz.add_axes([0.3, 0.73, 0.4, 0.05])
    
    # Create sliders
    sz_s1 = Slider(ax=axz_s1, label='s1 ', valmin=-10, valmax=10.0,
                  valfmt=' %1.3f', facecolor='#cc7000')
    sz_s2 = Slider(ax=axz_s2, label='s2 ', valmin=-2, valmax=5, 
                 valinit=1, valfmt='%1.3f ', facecolor='#cc7000')
    sz_t1 = Slider(ax=axz_t1, label='t1 ', valmin=-5, valmax=100.0,
                  valfmt=' %1.1f', facecolor='#cc7000')
    sz_t2 = Slider(ax=axz_t2, label='t2 ', valmin=-5, valmax=100, 
                 valinit=1, valfmt='%1.1f ', facecolor='#cc7000')
    
    f_Sz, = axz.plot(S_of_zeta(zeta, -4, s1, s2), zeta, label='S')
    f_zz, = axz.plot(theta_from_zeta(zeta, -4, s1, s2), zeta, label='theta')
    f_Tz, = axz.plot(T_of_zeta(zeta, -4, t1, t2), zeta, label='T')
    f_dz, = axz.plot(dif_of_zeta(zeta, -4, s1, s2, t1, t2), zeta, label='dif')
    # f_dpz, = axz.plot(dif_prime(theta, -4, s1, s2, t1, t2), zeta, label='dif_prime')
    axz.set_title('function vs zeta')
    axz.set_ylabel('zeta')
    axz.legend()
    
    def update_zeta(val):
        s1 = sz_s1.val
        s2 = sz_s2.val
        t1 = sz_t1.val
        t2 = sz_t2.val
        
        f_Sz.set_data(S_of_zeta(zeta, -4, s1, s2), zeta)
        f_zz.set_data(theta_from_zeta(zeta, -4, s1, s2), zeta)
        f_Tz.set_data(T_of_zeta(zeta, -4, t1, t2), zeta)
        f_dz.set_data(dif_of_zeta(zeta, -4, s1, s2, t1, t2), zeta)
        # f_dpz.set_data(dif_prime(theta, -4, s1, s2, t1, t2), zeta)
    
    sz_s1.on_changed(update_zeta)
    sz_s2.on_changed(update_zeta)
    sz_t1.on_changed(update_zeta)
    sz_t2.on_changed(update_zeta)























    