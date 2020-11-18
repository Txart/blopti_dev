# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import fipy as fp
from fipy.tools import numerix
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.close('all')

######################
# Comparison between theta- and h- based Boussinesq eq.
# Steps:
#   1) choose an artificial theta(h) and T(h)
#   2) Compute S(h) = d theta/ dh)
#   3) Compute S(theta)
#   4) Compute D(theta) = T/S
#   5) Solve Boussinesq eq in both forms and compare results


"""
Choice: theta(h) = exp(s0 + s1*h)
"""
#   1) theta(h) = exp(a + b*h); T(h) = t0 + exp(t1*h**t2)
#   2) S(h) = s1*exp(s0 + s1*h)
#   3) S(theta) = s1*theta
#   4) D(theta) = t0/s1 * exp(t1* ((ln(theta) -s0)/s1)**t2 - ln(theta))

nx = 300
dx = 1
dt = 10.

mesh = fp.Grid1D(nx=nx, dx=dx)

s0 = 0.1; s1 = 0.2
t0 = 1.; t1 = 0.01; t2 = 1.1

# IC
hini = 1.

h = fp.CellVariable(name="head", mesh=mesh, value=hini, hasOld=True)
h_implicit_source = fp.CellVariable(name="head_implicit_source", mesh=mesh, value=hini, hasOld=True)
h_convection = fp.CellVariable(name="head_convection", mesh=mesh, value=hini, hasOld=True)
theta = fp.CellVariable(name="theta", mesh=mesh, value=numerix.exp(s0 + s1*h.value), hasOld=True)

T = t0 * numerix.exp(t1 * h**t2)
S = s1 * numerix.exp(s0 + s1 * h) # and S_theta = s1 * theta
D = t0/s1 * numerix.exp(t1* ((numerix.log(theta) -s0)/s1)**t2)/ theta

# Plots
plotOpt = True
if plotOpt:
    fig, ax = plt.subplots(2,2)
    x = mesh.cellCenters.value[0]
    
    head = np.arange(0,10,0.1)
    th = np.exp(s0 + s1*head)
    trans = t0 * np.exp(t1 * head**t2)
    sto = s1 * np.exp(s0 + s1*head)
    diff_theta = t0 * np.exp(t1 * ((np.log(th)-s0)/s1)**t2) / (s1*th)
    diff_h = trans/sto
    
    if abs(diff_theta - diff_h).all() < 1e-5: print('D correctly calculated')
    
    ax[0,0].plot(head, th, label='theta'); ax[0,0].legend()
    ax[0,1].plot(diff_theta, th, label='D(theta)'); ax[0,1].legend()
    ax[1,0].plot(sto, head, label='S'); ax[1,0].legend()    
    ax[1,1].plot(trans, head, label='T'); ax[1,1].legend()

# BC
h_left = 1.; h_right = 0.
theta_left = numerix.exp(s0 + s1*h_left); theta_right = numerix.exp(s0 + s1*h_right)

h.constrain(h_left, where=mesh.facesLeft); h.constrain(h_right, where=mesh.facesRight)
h_implicit_source.constrain(h_left, where=mesh.facesLeft); h_implicit_source.constrain(h_right, where=mesh.facesRight)
h_convection.constrain(h_left, where=mesh.facesLeft); h_convection.constrain(h_right, where=mesh.facesRight)
theta.constrain(theta_left, where=mesh.facesLeft); theta.constrain(theta_right, where=mesh.facesRight)

# Boussinesq eq.

P = 0; ET = 0

eq_h_implicit_source = fp.TransientTerm(coeff=S) == fp.DiffusionTerm(coeff=T) +  P - ET + fp.ImplicitSourceTerm((S - S.old)/dt)
eq_h = fp.TransientTerm(coeff=S) == fp.DiffusionTerm(coeff=T) +  P - ET
# eq_h_convection = fp.TransientTerm() == fp.DiffusionTerm(coeff=T/S) +  P - ET + T/S**2 * S.faceGrad* fp.ExponentialConvectionTerm()
eq_theta = fp.TransientTerm() == fp.DiffusionTerm(coeff=D) + P - ET


#%%
"""
Solve several equations:
    - h version with implicit source term
    - h version without implicit source term
    - h version with S inside spatial derivative
    - theta version
"""
STEPS = 100
MAX_SWEEPS = 100
PLOT_EVERY = 10
for step in range(STEPS):
    
    h.updateOld() # also updates S_old
    h_implicit_source.updateOld()
    theta.updateOld()
    
    # h version of the eqn.
    res = 0.0
    for r in range(MAX_SWEEPS):
        resOld=res
        
        res = eq_h.sweep(var=h, dt=dt)
        
        if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
    
    # Implicit source version of the eqn.
    res = 0.0
    for r in range(MAX_SWEEPS):
        resOld=res
        
        res = eq_h_implicit_source.sweep(var=h_implicit_source, dt=dt)
        
        if abs(res - resOld) < 1e-7: break
    
    # # Convection version of the eqn.
    # res = 0.0
    # for r in range(MAX_SWEEPS):
    #     resOld=res
        
    #     res = eq_h_convection.sweep(var=h_convection, dt=dt)
        
    #     if abs(res - resOld) < 1e-7: break
    
    # theta version of the eqn.
    res = 0.0
    for r in range(MAX_SWEEPS):
        resOld=res
        
        res = eq_theta.sweep(var=theta, dt=dt)
        
        if abs(res - resOld) < 1e-7: break 
    
    if __name__ == '__main__' and step%PLOT_EVERY == 0:
        h_from_theta = (numerix.log(theta) -s0)/s1
        viewer = fp.Viewer(vars=(h, h_from_theta, h_implicit_source), datamin=0, datamax=2)
        viewer.plot()
        
#%%       
"""
Compare solutions for h and theta
"""
h_sol = h.value
h_implicit_source_sol = h_implicit_source.value
theta_sol = theta.value
theta_sol_from_h = np.exp(s0 + s1 * h_sol)  

if (abs(theta_sol - theta_sol_from_h) < 1e-2).all(): print('both versions of the eq give same result!')

