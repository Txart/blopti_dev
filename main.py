# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:45 2018

@author: L1817
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import argparse
import time
from pathlib import Path


import preprocess_data,  utilities, hydro_standard, hydro_utils, read


#%%
plt.close("all")

"""
Read general help on main.README.txt
"""

"""
Parse command-line arguments
"""
parser = argparse.ArgumentParser(description='Run hydro without any optimization.')

parser.add_argument('-d','--days', default=3, help='(int) Number of outermost iterations of the fipy solver, be it steadystate or transient. Default=10.', type=int)
parser.add_argument('-b','--nblocks', default=0, help='(int) Number of blocks to locate. Default=5.', type=int)
parser.add_argument('-n','--niter', default=1, help='(int) Number of repetitions of the whole computation. Default=10', type=int)
args = parser.parse_args()

DAYS = args.days
N_BLOCKS = args.nblocks
N_ITER = args.niter


"""
Read and preprocess data
"""
filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)

dem_rst_fn = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
can_rst_fn = Path(filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
peat_depth_rst_fn = Path(filenames_df[filenames_df.Content == 'peat_depth_raster'].Path.values[0])
blocks_fn = Path(filenames_df[filenames_df.Content == 'canal_blocks_raster'].Path.values[0])
sensor_loc_fn = Path(filenames_df[filenames_df.Content == 'sensor_locations'].Path.values[0])
params_fn = Path(filenames_df[filenames_df.Content == 'parameters'].Path.values[0])
WTD_folder = Path(filenames_df[filenames_df.Content == 'WTD_input_and_output_folder'].Path.values[0])
weather_fn = Path(filenames_df[filenames_df.Content == 'historic_precipitation'].Path.values[0])
# Choose smaller study area
STUDY_AREA = (0,-1), (0,-1)

wtd_old_fn = dem_rst_fn # not reading from previous wtd raster
can_arr, wtd_old , dem, peat_type_arr, peat_depth_arr, blocks_arr, sensor_loc_arr = preprocess_data.read_preprocess_rasters(STUDY_AREA, wtd_old_fn, can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn, blocks_fn, sensor_loc_fn)

sensor_loc_indices = utilities.get_sensor_loc_array_indices(sensor_loc_arr)

if 'CNM' and 'labelled_canals' and 'c_to_r_list' not in globals():
    labelled_canals = preprocess_data.label_canal_pixels(can_arr, dem)
    CNM, c_to_r_list = preprocess_data.gen_can_matrix_and_label_map(labelled_canals, dem)
 
built_block_positions = utilities.get_already_built_block_positions(blocks_arr, labelled_canals)

PARAMS_df = preprocess_data.read_params(params_fn)
BLOCK_HEIGHT = PARAMS_df.block_height[0]; CANAL_WATER_LEVEL = PARAMS_df.canal_water_level[0]
DIRI_BC = PARAMS_df.diri_bc[0]; HINI = PARAMS_df.hini[0]; P = PARAMS_df.P[0]
ET = PARAMS_df.ET[0]; TIMESTEP = PARAMS_df.timeStep[0]; KADJUST = PARAMS_df.Kadjust[0]

print(">>>>> WARNING, OVERWRITING PEAT DEPTH")
peat_depth_arr[peat_depth_arr < 2.] = 2.

# catchment mask
catchment_mask = np.ones(shape=dem.shape, dtype=bool)
catchment_mask[np.where(dem<-10)] = False # -99999.0 is current value of dem for nodata points.

# peel the dem. Only when dem is not surrounded by water
boundary_mask = utilities.peel_raster(dem, catchment_mask)
 
# after peeling, catchment_mask should only be the fruit:
# catchment_mask[boundary_mask] = False

# soil types and soil physical properties and soil depth:
peat_type_masked = peat_type_arr * catchment_mask
peat_bottom_elevation = - peat_depth_arr * catchment_mask # meters with respect to dem surface. Should be negative!
#

h_to_tra_and_C_dict, K = hydro_utils.peat_map_interp_functions(Kadjust=KADJUST) # Load peatmap soil types' physical properties dictionary

weather_stations_coordinates = [(100, 100), (200,200), (234, 142)] # TODO: change with coords from data
weather_station_mask, ws_mask_dict = preprocess_data.nearest_neighbors_mask_from_coordinates(dem.shape, weather_stations_coordinates)
# Then, weather_station_mask is to be used to mask the source term in the equation.

# Plot K
#import matplotlib.pyplot as plt
#plt.figure(); z = np.linspace(0.0, -20.0, 400); plt.plot(K,z); plt.title('K')
#soiltypes[soiltypes==255] = 0 # 255 is nodata value. 1 is water (useful for hydrology! Maybe, same treatment as canals).

#BOTTOM_ELE = -6.0 
#peat_bottom_elevation = np.ones(shape=dem.shape) * BOTTOM_ELE
#peat_bottom_elevation = peat_bottom_elevation*catchment_mask
tra_to_cut = hydro_utils.peat_map_h_to_tra(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = hydro_utils.peat_map_h_to_sto(soil_type_mask=peat_type_masked,
                                           gwt=peat_bottom_elevation, h_to_tra_and_C_dict=h_to_tra_and_C_dict)
sto_to_cut = sto_to_cut * catchment_mask.ravel()

srfcanlist =[dem[coords] for coords in c_to_r_list]

n_canals = len(c_to_r_list)


# HANDCRAFTED WATER LEVEL IN CANALS. CHANGE WITH MEASURED, IDEALLY.
oWTcanlist = [x - CANAL_WATER_LEVEL for x in srfcanlist]

include_already_built_blocks = True
hand_made_dams = False # compute performance of cherry-picked locations for dams.
quasi_random = False # Don't allow overlapping blocks
"""
MonteCarlo
"""
for i in range(0,N_ITER):
    
    if quasi_random == False or i==0: # Normal fully random block configurations
        damLocation = np.random.randint(1, n_canals, N_BLOCKS).tolist() # Generate random kvector. 0 is not a good position in c_to_r_list
    else:
        prohibited_node_list = [i for i,_ in enumerate(oWTcanlist[1:]) if oWTcanlist[1:][i] < wt_canals[1:][i]]      # [1:] is to take the 0th element out of the loop
        candidate_node_list = np.array([e for e in range(1, n_canals) if e not in prohibited_node_list]) # remove 0 from the range of possible canals
        damLocation = np.random.choice(candidate_node_list, size=N_BLOCKS)

    if hand_made_dams:
        # HAND-MADE RULE OF DAM POSITIONS TO ADD:
        hand_picked_dams = (11170, 10237, 10514, 2932, 4794, 8921, 4785, 5837, 7300, 6868) # rule-based approach
        hand_picked_dams = [6959, 901, 945, 9337, 10089, 7627, 1637, 7863, 7148, 7138, 3450, 1466, 420, 4608, 4303, 6908, 9405, 8289, 7343, 2534, 9349, 6272, 8770, 2430, 2654, 6225, 11152, 118, 4013, 3381, 6804, 6614, 7840, 9839, 5627, 3819, 7971, 402, 6974, 7584, 3188, 8316, 1521, 856, 770, 6504, 707, 5478, 5512, 1732, 3635, 1902, 2912, 9220, 1496, 11003, 8371, 10393, 2293, 4901, 5892, 6110, 2118, 4485, 6379, 10300, 6451, 5619, 9871, 9502, 1737, 4368, 7290, 9071, 11222, 3085, 2013, 5226, 597, 5038]
        damLocation = hand_picked_dams
    
    if include_already_built_blocks:
        damLocation = damLocation + list(built_block_positions)
    
    wt_canals = utilities.place_dams(oWTcanlist, srfcanlist, BLOCK_HEIGHT, damLocation, CNM)
    """
    #########################################
                    HYDROLOGY
    #########################################
    """
    ny, nx = dem.shape
    dx = 1.; dy = 1. # metres per pixel  (Actually, pixel size is 100m x 100m, so all units have to be converted afterwards)
    
    boundary_arr = boundary_mask * (dem - DIRI_BC) # constant Dirichlet value in the boundaries
    
    # P = read.read_precipitation()
    P = [0.0] * DAYS 
    # ET = ET * np.ones(shape=P.shape)
    ET = [0.0] * DAYS
    
    ele = dem * catchment_mask
    
    # Get a pickled phi solution (not ele-phi!) computed before without blocks, independently,
    # and use it as initial condition to improve convergence time of the new solution
    retrieve_transient_phi_sol_from_pickled = False
    if retrieve_transient_phi_sol_from_pickled:
        with open(r"pickled/transient_phi_sol.pkl", 'r') as f:
            phi_ini = pickle.load(f)
        print("transient phi solution loaded as initial condition")
        
    else:
        phi_ini = ele + HINI #initial h (gwl) in the compartment.
        phi_ini = phi_ini * catchment_mask
           
    wt_canal_arr = np.zeros((ny,nx)) # (nx,ny) array with wt canal height in corresponding nodes
    for canaln, coords in enumerate(c_to_r_list):
        if canaln == 0: 
            continue # because c_to_r_list begins at 1
        wt_canal_arr[coords] = wt_canals[canaln]
    
    
    wtd = hydro_standard.hydrology('transient', nx, ny, dx, dy, DAYS, ele, phi_ini, catchment_mask, wt_canal_arr, boundary_arr,
                                                      peat_type_mask=peat_type_masked, httd=h_to_tra_and_C_dict, tra_to_cut=tra_to_cut, sto_to_cut=sto_to_cut,
                                                      diri_bc=None, neumann_bc = 0., plotOpt=True, remove_ponding_water=True,
                                                      P=P, ET=ET, dt=TIMESTEP)
    
    
        
        
        
    
    # water_blocked_canals = sum(np.subtract(wt_canals[1:], oWTcanlist[1:]))
    
    # cum_Vdp_nodams = 21088.453521509597
    # print('dry_peat_volume(%) = ', dry_peat_volume/cum_Vdp_nodams * 100. , '\n',
    #       'water_blocked_canals = ', water_blocked_canals)

    """
    Final printings
    """
    # if quasi_random == True:
    #     fname = r'output/results_mc_quasi_3.txt'
    # else:
    #     fname = r'output/results_mc_3_cumulative.txt'
    # if N_ITER > 20:
        
    #     with open(fname, 'a') as output_file:
    #         output_file.write(
    #                             "\n" + str(i) + "    " + str(dry_peat_volume) + "    "
    #                             + str(N_BLOCKS) + "    " + str(N_ITER) + "    " + str(DAYS) + "    "
    #                             + str(time.ctime()) + "    " + str(water_blocked_canals)
    #                           )

#%%
# Translate WT to CO2 and subsidence
# def CO2(wtd, peat_type):

landcover_fn = Path(filenames_df[filenames_df.Content == 'landcover'].Path.values[0])
lc = preprocess_data.read_preprocess_landcover(STUDY_AREA, landcover_fn)
# Get coefficients and compute CO2
co2_mult_coef, co2_add_coef = utilities.map_lc_number_to_lc_coef_in_co2_emission_formula(lc)
co2 = utilities.compute_co2_from_WTD(wtd, co2_mult_coef, co2_add_coef)
# Get coefs and compute subsidence
subsi_mult_coef, subsi_add_coef = utilities.map_lc_number_to_lc_coef_in_subsidence_formula(lc)
subsi = utilities.compute_subsi_from_WTD(wtd, subsi_mult_coef, subsi_add_coef) #m/day

# output as a single multiband raster
multiband = np.array([wtd, co2, subsi])

utilities.write_raster_multiband(3, multiband, STUDY_AREA, out_filename='output/dem_co2_subsi.tif', ref_filename=dem_rst_fn)    
    
"""
Save WTD data if simulating a year
"""
# fname = r'output/wtd_year_' + str(N_BLOCKS) + '.txt'
# if DAYS > 300:
#    with open(fname, 'a') as output_file:
#        output_file.write("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" +
#                              str(time.ctime()) + " nblocks = " + str(N_BLOCKS) + " ET = " + str(ET[0]) +
#                              '\n' + 'drained notdrained mean'
#                              )
#        for i in range(len(wt_track_drained)): 
#            output_file.write( "\n" + str(wt_track_drained[i]) + " " + str(wt_track_notdrained[i]) + " " + str(avg_wt_over_time[i]))

# plt.figure()
# plt.plot(list(range(0,DAYS)), wt_track_drained, label='close to drained')
# plt.plot(list(range(0,DAYS)), wt_track_notdrained, label='away from drained')
# plt.plot(list(range(0,DAYS)), avg_wt_over_time, label='average')
# plt.xlabel('time(days)'); plt.ylabel('WTD (m)')
# plt.legend()
# plt.show()
