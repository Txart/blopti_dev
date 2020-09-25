# -*- coding: utf-8 -*-
"""
CHECK ACCURACY:
    Compares Forward Euler, RK4, FiPy
    Compares Forward Euler and RK4 with decreasing timesteps

IMPORTANT NOTE: 
    In order to run this, I had to manually change the solver in hydro_calibration
    That is, to choose either Runge Kutta 4th or Forward Euler, go to
    hydro_calibration.hydro_1d_chebyshev and uncomment the line of the solver
    
    Also, you'll see some red crosses on the left. Most of the variables are not
    defined here, but on calibration_1d.py. Run that first and once those variables
    are in memory, run this.
    
RESULTS:
    - I couldn't compare FiPy despite the effort. It gave bad results. (Not only many
    matrix is excatly singular, but also very different results from the other two)
    - There's at most a 0.05 m difference in h for the different dt evaluated for both
    RK4 and Forward Euler. dts in range: dt=1e-2... 1e-6. Take a look at figures in 
    PhD/ArticleII
    
    
Created on Wed Sep 16 10:02:16 2020

@author: 03125327
"""
import time
import matplotlib.pyplot as plt
import numpy as np
#%%
rnd_params = np.random.rand(20,5)*2

#%%
# script file
# Compare fipy, RK4, Euler forward
start_time = time.time()
simu_rk4 = []
for params in rnd_params:
    dt = 1e-3
    simulated_wtd = hydro_calibration.hydro_1d_chebyshev(theta_ini, nx-1, dx, dt,
                                                         params, ndays, sensor_locations,
                                                         theta_boundary_values_left,
                                                         theta_boundary_values_right,
                                                         precip, evapotra, ele)
    simu_rk4.append(simulated_wtd)

print(f"rk4 time: {time.time() - start_time}")



#%%
start_time = time.time()
simu_fipy = []
for params in rnd_params:
    dt=1
    try:
        simulated_wtd = hydro_calibration.hydro_1d_fipy(theta_ini, nx, dx, dt,
                                                         params, ndays, sensor_locations,
                                                         theta_boundary_values_left,
                                                         theta_boundary_values_right,
                                                         precip, evapotra, ele)
    except:
        simulated_wtd = np.ones(ndays) * np.nan
    simu_fipy.append(simulated_wtd)

print(f"fipy time: {time.time() - start_time}")



#%%
npse = np.array(simu_euler); npsr = np.array(simu_rk4); npfp = np.array(simu_fipy)
    
dif_eu_rk4 = np.sum(npse**2 - npsr**2, axis=1)
dif_eu_fipy = np.sum(npse**2 - npfp**2, axis=1)

#%%
for i,_ in enumerate(rnd_params):
    plt.figure(i)
    plt.plot(simu_euler[i], 'x', label="Euler")
    plt.plot(simu_rk4[i], 'x', label="RK4")
    plt.plot(simu_fipy[i], 'x', label="fipy")
    plt.legend()
    
    
plt.show()


#%%
simu_euler = np.array(simu_euler)
simu_rk4 = np.array(simu_rk4)
simu_fipy = np.array(simu_fipy)

#%%
# check accuracy
acc_values = 10
dts = np.linspace(1e-2, 1e-6, acc_values)
params = [1.0, 2.1, 2.0, 1.0, 0.3]
simu_eu_acc = []
for dt in dts:
    simulated_wtd = hydro_calibration.hydro_1d_chebyshev(theta_ini, nx-1, dx, dt,
                                                         params, ndays, sensor_locations,
                                                         theta_boundary_values_left,
                                                         theta_boundary_values_right,
                                                         precip, evapotra, ele)
    simu_eu_acc.append(simulated_wtd)
#%%
# plot accuracy    
plt.figure('accuracy')
cmap_eu = plt.cm.Purples
cmap_rk = plt.cm.Oranges
# extract all colors from the .jet map
cmaplist_eu = [cmap_eu(int(i)) for i in np.linspace(0,255,acc_values)]
cmaplist_rk = [cmap_rk(int(i)) for i in np.linspace(0,255,acc_values)]
for i,dt in enumerate(dts):
    plt.plot(simu_rk_acc[i], color=cmaplist_rk[i])
    plt.plot(simu_eu_acc[i], color=cmaplist_eu[i])

plt.xlabel('ndays')
plt.ylabel('h')
plt.title('Runge-Kutta: orange, Fwd. Euler: blue. Light -> Dark: dt=0.01 -> dt=1e-6. params = [1.0, 2.1, 2.0, 1.0, 0.3]')    
    
    
    
    
    