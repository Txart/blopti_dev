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
parser.add_argument('--ncdays', default=2, help='(int) Minimum value of consecutive days shown. Default: 2', type=int)

args = parser.parse_args()
 
only_P_sensors = args.onlyp
plotOpt = args.plot
ncdays = args.ncdays


if only_P_sensors > 1 or plotOpt > 1:
    raise ValueError('arguments --onlyp and --plot have to be either 0 or 1.')

#%%
"""
 Read and choose data
"""
# read, preprocess data
fn_weather_data = Path('data/weather_station_historic_data.xlsx')
dfs_by_transects = get_data.main(fn_weather_data)

if only_P_sensors:
    ps = [k for k in dfs_by_transects.keys() if 'P0' in k]
    dfs_by_transects = {x: dfs_by_transects[x] for x in ps}

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
        
        # plot Precip and ET
        ax2 = ax.twinx()
        ax2.margins(x=0)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(4)) # make only 4 ticks in the x axis
        ax2.xaxis.set_minor_locator(ticker.MaxNLocator(4*5))
        ax2.set_ylabel('P-ET (mm/day)')
        ax2.set_ylim([-50, 100])
        p_minus_et = df_sorted['P'] - df_sorted['ET']
        ax2.bar(x=df_sorted['Date'], height=p_minus_et, color='blue', alpha=0.3, width=1)
        
        for col_name in df_sorted: # iterate dataframe by columns
             
            if col_name not in col_labels_not_to_plot:
                ax.plot(df_sorted['Date'], df_sorted[col_name], 'o', label=col_name)
    
        ax.legend()
        

#%%
"""
 Print consecutive days
"""
from itertools import groupby
from operator import itemgetter


for transect_name, transect_df in dfs_by_transects.items():
    print(f'Transect name: {transect_name}')
    daylist = list(transect_df['julian_day'])
    
    for k,g in groupby(enumerate(daylist), lambda ix: ix[0] - ix[1]):
        printable = list(map(itemgetter(1),g))
        if len(printable) >= ncdays:
            print(f'>>>>> {printable}')




#%%
def f1():
    for i in range(0,3):
        try:
            raise ValueError
        except:
            print('some errorin try')
            return 2
        else:
            print('else!')
            return 3

    return 4












