# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:59:07 2020

@author: slauniai

Estimate daily ET from meteorological station data in absense of radiation data:
    1) estimate net radiation balance in absense of direct measurements
    2) compute daily reference ETo
    3) compute actual evaporation as ETa = Kc x ETo

    The crop coefficient Kc should be determined against ET measurements from
    similar ecosystem; it is function of LAI, plant water use traits, albedo, soil moisture etc.

Step 1) can be omitted or modified if meteorological station provides radiation data or records sunshine hours

Tested part 1) with FAO -reference Example 16
 
    
REFERENCE:    
ETo = "reference crop ET", aka potential evapotranspiration. Expresses the evaporating power of the atmosphere
ETa = Kc x ETo, where Kc is crop coefficient (or fraction of ETo that is realized)

Needed input:
    - Common data: day
    - Site data: latitude, elevation
    - Meteo data: Tmin, Tmax, average T, U(winspeed), ea(vapor pressure)
    - 
    - Params: Kc, krs(parameter ranging 0.16 ... 0.19)

Params DEFINED by FAO:
    - albedo = 0.23
"""
import numpy as np
import evapotranspiration_fao as et

# Some sample and defect parameter values 

jday = 100.0 # julian day. 100 = 10th of April
latitude = -2.216105 # decimal deg. Palangka Raya (Central Kalimantan), site for Hirano et al.'s forest
elev = 2.0 # m

albedo = 0.23 # definition. For reference crop
# albedo = 0.09 # From Hirano et al.

# meteorological data
Tmin = 25.0 # degC, April, Borneo
Tmax = 30 # degC
Tave = 27.0
U = 2.2 # Winspeed at 2m [ms-1]
# ea = 2.85 # vapor pressure at 2m [kPa]
# vapour pressure from relative humidity:
RH = 79.0 # %
es, _ =  et.saturation_vapor_pressure(Tave)
ea = (1 - RH / 100.0) * es

# Params
Kc = 0.9 # [-] crop coefficient, Kc = ETa / ETo
KRS = 0.19

air_pressure = et.pressure_from_altitude(elev) # Computes air pressure in the station given sea level air pressure

def compute_ET(jday, Tmax, Tave, Tmin, RH, U, print_res=False):
    """
    Main function to return ET

    Parameters
    ----------
    jday : int
        julian day
    print_res : bool, optional
        Whether or not to print output. The default is False.

    Returns
    -------
    ETa : float
        ETa = Kc x ETo, where Kc is crop coefficient (or fraction of ETo that is realized)
    ETo : float
        ETo = "reference crop ET", aka potential evapotranspiration. Expresses the evaporating power of the atmosphere
    Rn : float
        Daily net radiation of reference crop
    Rs : float
        Daily solar radiation
    Rnl : float
        Daily net longwave radiation

    """
    """
     1) net radiation balance [MJ m-2 d-1]
    """
    # daily solar radiation and daily clear-sky solar radiation estimated from met. station data
    Rs, Rso = et.fao_shortwave_radiation(latitude, jday, Tmin, Tmax, krs=KRS)
    
    # daily net longwave radiation 
    Rnl = et.fao_net_longwave_radiation(Tmin, Tmax, ea, Rs, Rso)
    
    # daily net radiation of reference crop
    Rn = (1. - albedo) * Rs - Rnl
    
    """
     2) daily reference ETo [mm d-1]
    """
    ETo = et.fao_reference_et(Rn, Tave, U, ea, P=air_pressure)
    
    """
     3) actual ET [mm d-1]
    """
    ETa = et.actual_evapotranspiration(ETo, Kc)
    
    if print_res:
        print('*** ETa = %.2f mm d-1, ETo = %.2f mm d-1 ***' %(ETa, ETo))
        print('*** Annual ETa = %.2f mm y-1, Annual ETo = %.2f mm y-1 ***' %(ETa*365, ETo*365))
        print('*** Rn = %.2f MJ m-2 d-1, Rs = %.2f MJ m-2 d-1, Rnl = %.2f MJ m-2 d-1 ***' %(Rn, Rs, Rnl))
        print('*** Annual Rn = %.2f GJ m-2 y-1, Annual Rs = %.2f GJ m-2 y-1, Annual Rnl = %.2f GJ m-2 y-1 ***' %(Rn*.365, Rs*.365, Rnl*.365))

    
    return ETa, ETo, Rn, Rs, Rnl



#%%
# Compute year averages of all the quantities
if __name__ == '__main__':


    # Temperature, humidity and windspeed monthly averages in year 2009
    # taken from: https://www.worldweatheronline.com/palangkaraya-weather-averages/kalimantan-tengah/id.aspx
    temperature= ([[31,26,23],]*31 + # [Max, Mean, Min]; Jan
                   [[31,26,23],]*28 + # Feb
                   [[30,26,23],]*31 + # Mar
                   [[31,26,23],]*30 + # apr
                   [[32,27,23],]*31 + # May
                   [[32,27,23],]*30 + # Jun
                   [[31,26,22],]*31 + # Jul
                   [[32,27,23],]*31 + # Aug
                   [[33,28,23],]*30 + # Sep
                   [[31,26,23],]*31 + # Oct
                   [[30,26,23],]*30 + # Nov
                   [[30,26,23],]*31 # Dec
                   )
    
    humidity = ([86]*31 + [84]*28 + [90]*31 + [88]*30 + [84]*31 + [82]*30 + # daily rel humidity
                     [82]*31 + [75]*31 + [73]*30 + [84]*31 + [87]*30 + [90]*31) 
    windspeed = ([4.1]*31 + [4.2]*28 + [3.2]*31 + [3.1]*30 + [3.1]*31 + [3.7]*30 + # daily  windspeed [km h-1]
                     [4.4]*31 + [5.2]*31 + [6.1]*30 + [4.0]*31 + [3.7]*30 + [3.2]*31)
    windspeed = np.array(windspeed)*1000/3600 #[m/s]
                 
    res = np.array([0,0,0,0,0])
    
    for jday in range(0, 365):
        Tmax = temperature[jday][0]; Tmean = temperature[jday][1]; Tmin = temperature[jday][2]
        RH = humidity[jday]; U = windspeed[jday]
        s = np.array(compute_ET(jday+1, Tmax=Tmax, Tave=Tmean, Tmin=Tmin, RH=RH, U=U))
        res = res + s
    
    print(f"Annual ETa = {res[0]}")
    print(f"Annual ETo = {res[1]}")
    print(f"Annual Rn [GJ m-2 y-1] = {res[2]/1000}")
    print(f"Annual Rs [GJ m-2 y-1] = {res[3]/1000}")
    print(f"Annual Rnl [GJ m-2 y-1] = {res[4]/1000}")
    
    #%%
    # Plot Rn and ET over a year
    import matplotlib.pyplot as plt
    
    Kc = 0.9
    KRS = 0.19
    
    
    year_ETa = []
    year_Rn = []
    for jday in range(0, 365):
        Tmax = temperature[jday][0]; Tmean = temperature[jday][1]; Tmin = temperature[jday][2]
        RH = humidity[jday]; U = windspeed[jday]
        ETa, _, Rn, _, _ = compute_ET(jday+1, Tmax=Tmax, Tave=Tmean, Tmin=Tmin, RH=RH, U=U)
        year_ETa.append(ETa); year_Rn.append(Rn)
        
    plt.figure()
    plt.plot(list(range(0,365)), year_ETa, label='ETa [mm d-1]')
    plt.plot(list(range(0,365)), year_Rn, label = 'Rn [MJ m-2 d-1]')
    plt.xlabel('julian days')
    
    plt.legend()
    #%%
    # data sources and others
    """
     Rn and ET averages for years(2004-2007) from Hirano et al. 2015 Evapotranspiration of tropical peat swamp forests
     T timeseries from 
         Year:   forest type*, Rn [GJ m-2 yr-1], ET[mm yr-1] 
         ----   
        2004:   UF, 4.78, 1634
                DF, 4.71, 1529
                DB, 4.53, 1359
        2005:   UF, 4.77, 1648
                DF, 4.79, 1611
                DB, 4.46, 1404
        2006:   UF, 4.58, 1566
                DF, 4.41, 1401
                DB, 4.25, 1277
        2007:   UF, 4.92, 1695
                DF, 4.87, 1671
                DB, 4.68, 1454
        Mean:   UF, 4.76, 1636
                DF, 4.70, 1553
                DB, 4.48, 1374
                
        * UF: Undrained Forest; DF = Drained Forest; DB = Drained Burnt forest
    
    Temperature, humidity and windspeed averages from: https://www.worldweatheronline.com/palangkaraya-weather-averages/kalimantan-tengah/id.aspx
        Tavg = 27  
        Tmax = 30
        Tmin = 25
        rel. hum. = 79%
        windspeed = 8km/h = 2.22 m/s
    
    Fitting:
    1) First I fit the KRS parameter with Rn values. This gives KRS=0.35 for UF and DF; KRS=0.32 for DB
    2) Then I fit the Kc parameter with the ET values. This gives Kc=0.62 for UF; Kc=0.59 for DF; Kc=0.54 for DB
    
        ****** Carefull! This gives too high KRS values!!
    """