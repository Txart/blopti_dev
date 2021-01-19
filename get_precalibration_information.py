# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:14:15 2020

@author: 03125327
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path

import get_data

#%%
"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Get information for calibration')
 
parser.add_argument('--onlyp', default=1, help='1 or 0. Choose 1 to operate only with P sensors (not DOSANxxx or DAYUNxxx). Default: 1.', type=bool)
parser.add_argument('--plot', default=1, help='1 or 0. Choose 1 to plot historic sensor data. Default: 1', type=bool)
parser.add_argument('--ncdays', default=10, help='(int) Minimum value of consecutive days shown. Default: 2', type=int)
parser.add_argument('--drydown', default =1, help='1 or 0. Choose 1 to print only consecutive days with lowering water table. Default:1', type=bool)

args = parser.parse_args()
 
only_P_sensors = args.onlyp
plotOpt = args.plot
ncdays = args.ncdays
is_drydown = args.drydown


if only_P_sensors > 1 or plotOpt > 1:
    raise ValueError('arguments --onlyp and --plot have to be either 0 or 1.')

#%%
"""
 Read and choose data
"""
# read, preprocess data
fn_weather_data = Path('data/weather_station_historic_data.xlsx')
fn_wtd_data = Path('data/historic_wtd_data_18-1-2021.xlsx')
dfs_by_transects = get_data.main(fn_weather_data, fn_wtd_data, api_call=False)

if only_P_sensors:
    ps = [k for k in dfs_by_transects.keys() if 'P0' in k]
    dfs_by_transects = {x: dfs_by_transects[x] for x in ps}
    
# Take out some bad data:
dfs_by_transects.pop('P014')
dfs_by_transects.pop('P020')


#%%
"""
Print ncdays consecutive drydown days, i.e.,
consecutive days that follow  conditions 1), 2) AND 2):
    1) water level on the sensor furthest from the canal is lower than the day before
    2) Days are consecutive days in the calendar
    3) There are more than ncdays consecutive days
"""
for transect_name, transect_df in dfs_by_transects.items():
    print(f'Transect name: {transect_name}')
    daylist = transect_df['julian_day'].to_numpy()
    wtlist = transect_df['sensor_1'].to_numpy()
    
    day_number_dif_with_previous_day = np.ediff1d(daylist)
    wt_dif_with_previous_day = np.ediff1d(wtlist)
    
    consecutive_days_mask = day_number_dif_with_previous_day == 1
    if is_drydown:
        drydown_mask = wt_dif_with_previous_day <= 0 # water table goes down condition
    else:
        drydown_mask = np.ones(shape=consecutive_days_mask.shape, dtype=bool)
    combined_mask = np.logical_and(drydown_mask, consecutive_days_mask)
    
    
    set_of_good_days = set()
    for i,good_day in enumerate(combined_mask):
        if good_day:
            set_of_good_days.update([daylist[i], daylist[i+1]])
        elif not good_day:
            if len(set_of_good_days) >= ncdays:
                print(f'>>>>>{sorted(list(set_of_good_days))}')
            set_of_good_days = set()
    
    # Check if the last is True!
    if len(set_of_good_days) >= ncdays:
        print(f'>>>>>{list(set_of_good_days)}')
            
        
    
    
#%%
"""
 Plot data by sensor pairs
"""   
col_labels_not_to_plot = ['julian_day', 'Date', 'T_ave', 'T_min', 'T_max', 'P', 'ET', 'rel_hum', 'windspeed', 'air_pressure']

if plotOpt:
    print('Plotting transect historical data... \n')
    
    for transect_name, transect_df in dfs_by_transects.items():
        
        df_sorted = transect_df.sort_index()
        
        fig, ax = plt.subplots(num=transect_name)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4)) # make only 4 ticks in the x axis
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(4*5))
        ax.set_ylabel('WTD (m)')
        ax.set_ylim([-2,.50])
        for col_name in df_sorted: # iterate dataframe by columns
            if col_name not in col_labels_not_to_plot: # i.e., plot only sensor values
                ax.plot(df_sorted['Date'], df_sorted[col_name], 'o', label=col_name)
        
        # plot Precip and ET
        ax2 = ax.twinx()
        ax2.margins(x=0)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(4)) # make only 4 ticks in the x axis
        ax2.xaxis.set_minor_locator(ticker.MaxNLocator(4*5))
        ax2.set_ylabel('P-ET (mm/day)')
        ax2.set_ylim([-50, 100])
        p_minus_et = (df_sorted['P'] - df_sorted['ET']) * 1000 #m/day -> mm/day
        ax2.bar(x=df_sorted['Date'], height=p_minus_et, color='blue', alpha=0.3, width=1)
        
        ax.legend()
        
    plt.show()
    
        











