# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:18:31 2020

@author: 03125327
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import csv
import get_data

csv.field_size_limit(100000000) # increase field size limit of csv. The column containing the vectors is otherwise too big!


#%% Read data
filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)
cwl_est_out_fn = Path(filenames_df[filenames_df.Content == 'cwl_estimation_output'].Path.values[0])

cwl_reference_df = pd.read_csv(cwl_est_out_fn, engine='python', sep=';', header=None, names=['day','cwl_sensors', 'cwl_all'])

cwl_ref_sensors = np.vstack(cwl_reference_df['cwl_sensors'].apply(ast.literal_eval).values) *100 #in cm
cwl_ref_all = cwl_reference_df['cwl_all'].apply(ast.literal_eval).values

    
#%% Get daily sensor data 
import requests
from io import StringIO

def get_wt_data(measurement='wt'):
    # TODO: STANDARIZE TIME AND DATE INPUT. cHECK DASHBOARD API TUTORIAL!
    url_base = "https://dashboard.sustainabilitytech.com/api/"
    api_key = '61b8b35c1a42075c2f2de046ffe1975c908bc41355bda38f15efcf9cb9e7/'
    version = 'v1/'
    format_csv = 'csv/'
    if measurement == 'wt':
        collection = 'measurement/module[waterTable]/'
        created = 'created[>10-03-2020&<12-03-2020]'
        url = url_base + api_key + version + format_csv + collection + created
    elif measurement=='sensors':
        collection = 'sensor/projectId[5c6a0f20868636437eaab987],module[waterTable]/label=1/undefined'
        url = url_base + api_key + version + format_csv + collection

    response = requests.request(method='POST', url=url)
    data = response.text
    strdata = StringIO(data)
    return pd.read_csv(strdata)

cwl_sensor_df = get_wt_data(measurement='wt')
sensor_data = get_wt_data(measurement='sensors')
# TODO: select only relevant data!
cwl_sensors = cwl_sensor_df.convertedValue[0:4].values

#%% Compare daily sensor data with precomputed cwl and pick the best!

cwl_sensor_vector = np.ones((len(cwl_ref_sensors),1))*cwl_sensors
cwl_diff = np.add(cwl_ref_sensors, -cwl_sensor_vector) # simple vector difference
cwl_dist = np.linalg.norm(cwl_diff, axis=1) # least squares

best_match = cwl_dist.argmin() # minimum least squares

cwl_of_best_match = np.array(cwl_ref_all[best_match])

    
    