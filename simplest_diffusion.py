# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:33:17 2020

@author: 03125327
"""


from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

nx = 50
dx = 1.
mesh = Grid1D(nx=nx, dx=dx)

phi = CellVariable(name='sol var', mesh=mesh, value=0., hasOld=False)

D = 1.

valueleft = 1
valueright = 0

phi.constrain(valueleft, mesh.facesLeft)
phi.constrain(valueright, mesh.facesRight)

eq = TransientTerm() == DiffusionTerm(coeff=D)

timeStepDuration = 1.
steps = 10

MAX_SWEEPS = 100
for step in range(steps):
    res = 0.0
    # phi.updateOld()
    for r in range(MAX_SWEEPS):
        resOld=res
   
        res = eq.sweep(var=phi, dt=timeStepDuration)
        
        if abs(res - resOld) < 1e-7: break # it has reached to the solution of the linear system
    
    if __name__ == '__main__':
         viewer = Viewer(vars=phi, datamin=0., datamax=1.)
         viewer.plot()