# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:22:00 2018

@author: L1817
"""

import numpy as np
import copy
import random
import scipy.signal

#%%

def get_already_built_block_positions(blocks_arr, labelled_canals):
    labelled_blocks = blocks_arr * labelled_canals
    labelled_blocks = labelled_blocks.astype(dtype=int)
    return tuple(labelled_blocks[labelled_blocks.nonzero()].tolist())

def get_sensor_loc_array_indices(sensor_loc_arr):
    locs = np.array(sensor_loc_arr.nonzero())
    loc_pairs = locs.transpose()
    return loc_pairs.tolist()
    

def peel_raster(raster, catchment_mask):
    """
    Given a raster and a mask, gets the "peeling" or "shell" of the raster. (Peeling here are points within the raster)
    Input:
        - raster: 2dimensional nparray. Raster to be peeled. The peeling is part of the raster.
        - catchment_mask: 2dim nparray of same size as raster. This is the fruit in the peeling.
    Output:
        - peeling_mask: boolean nparray. Tells where the peeling is.

    """
    # catchment mask boundaries by convolution
    conv_double = np.array([[0,1,1,1,0],
                            [1,1,1,1,1],
                            [1,1,0,1,1],
                            [1,1,1,1,1],
                            [0,1,1,1,0]])
    bound_double = scipy.signal.convolve2d(catchment_mask, conv_double, boundary='fill', fillvalue=False)
    peeling_mask = np.ones(shape=catchment_mask.shape, dtype=bool)
    peeling_mask[bound_double[2:-2,2:-2]==0] = False; peeling_mask[bound_double[2:-2,2:-2]==20] = False
    
    peeling_mask = (catchment_mask*peeling_mask) > 0

    return peeling_mask


# NEW 23.11. ONE STEP OR MOVEMENT IN THE SIMULATED ANNEALING.
def switch_one_dam(oWTcanals, surface_canals, currentWTcanals, block_height, dams_location, n_canals, CNM):
    """
        Randomly chooses which damn to take out and where to put it again.
    Computes which are "prohibited nodes", where the original water level of the canal is
    lower than the current (i.e., has been affected by the placement of a dam), before making the
    otherwise random decision of locating the dam.
    
    OUTPUT
        - new_wt_canal: wl in canal after removing old and placing a new dam.
    """
        
    # Select dam to add
    # Check for prohibited nodes. Minor problem: those points affected by the dam which will be soon removed are also prohibited
    prohibited_node_list = [i for i,_ in enumerate(oWTcanals) if oWTcanals[i] < currentWTcanals[i]]
    candidate_node_list = [e for e in range(0, n_canals) if e not in prohibited_node_list]
    random.shuffle(candidate_node_list) # Happens in-place.
    dam_to_add = candidate_node_list[0]
    
    # Select dam to remove
    random.shuffle(dams_location) # Shuffle to select which canal to remove. Happens in place.
    dam_to_remove = dams_location[0]
    
    dams_location.remove(dam_to_remove)
    dams_location.append(dam_to_add)
    
    # Compute new wt in canals with this configuration of dams
    new_wt_canal = place_dams(oWTcanals, surface_canals, block_height, dams_location, CNM)
    
    return new_wt_canal
    
    
    
    
    
    

def PeatV_weight_calc(canal_mask):
    """ Computes weights (associated to canal mask) needed for peat volume compt.
    
    input: canal_mask -- np.array of dim (nx,ny). 0s where canals or outside catchment, 1s in the rest.
    
    
    output: np.array of dim (nx,ny) with weights to compute energy of sim anneal
    """
    
    xdim = canal_mask.shape[0]
    ydim = canal_mask.shape[1] 
    
    # Auxiliary array of ones and zeros.
    arr_of_ones = np.zeros((xdim+2,ydim+2)) # Extra rows and cols of zeros to deal with boundary cases
    arr_of_ones[1:-1,1:-1] = canal_mask # mask
    
    # Weights array
    weights = np.ones((xdim,ydim)) 
    
    # Returns number of non-zero 0th order nearest neighbours
    def nn_squares_sum(arr, row, i):
        nsquares = 0
        if ((arr[row,i] + arr[row-1,i] + arr[row-1,i-1] + arr[row,i-1]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row,i-1] + arr[row+1,i-1] + arr[row+1,i]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row+1,i] + arr[row+1,i+1] + arr[row,i+1]) == 4):
            nsquares += 1
        if ((arr[row,i] + arr[row,i+1] + arr[row-1,i+1] + arr[row-1,i]) == 4):
            nsquares += 1
        return nsquares
    
    
    for j, row in enumerate(arr_of_ones[1:-1, 1:-1]):
        for i, _ in enumerate(row):
            weights[j,i] = nn_squares_sum(arr_of_ones, j+1, i+1)
    
    return weights

def PeatVolume(weights, Z):
    """Computation of dry peat volume. Energy for the simulated annealing.
    INPUT:
        - weights: weights as computed from nn_squares_sum function
        - Z: array of with values = surface dem elevation - wt
    OUTPUT:
        - Dry peat volume. Units: ha x m. The ha thing is coming from the pixel size being 100x100m and dx=dy=1.
        On the other hand, water table depth is in m.
    """
    
    # This is good code for the future.
#    sur = np.multiply(surface, weights) # element-wise multiplication
#    gwt = np.multiply(gwt, weights) # element-wise multiplication
#    gehi_sur = np.sum(sur) # element-wise sum
#    gehi_gwt = np.sum(gwt)
    zet = np.multiply(Z, weights) # all operations linear, so same as doing them with Z=surface-gwt
    z_sum = np.sum(zet)
    
    #Carefull with gwt over surface!
#    dry_peat_volume = .25 * (ygrid[1]-ygrid[0])*(xgrid[1]-xgrid[0]) * (gehi_sur - gehi_gwt)

    dry_peat_volume = .25 * z_sum
    
    return dry_peat_volume

def print_time_in_mins(time):
    if time >60 and time <3600:
        print("Time spent: ", time/60.0, "minutes")
    elif time >3600:
        print("Time spent: ", time/60.0, "hours")
    else:
        print("Time spent: ", time, "seconds")
        
   
def place_dams(originalWT, srfc, block_height, dams_to_add, CNM):
    """ Takes original water level in canals and list of nodes where to put blocks. Returns updated water level in canals.
    
    Input:
        - originalWT: list. Original water level in canals.
        - srfc: list. DEM value at canals.
        - block_height: float. Determines the new value of the water level as: new_value = surface[add_can] - block_height.
        - dams_to_add: list of ints. positions of dam to add.
        - CNM: propagation or canal adjacency (sparse) matrix.
        
    Output:
        - wt: list. Updated water level in canals.
    """
    
    def addDam(wt, surface, block_height, add_dam, CNM): 
        """ Gets a single canal label and returns the updated wt corresponding to building a dam in that canal
        """
        add_height = surface[add_dam] - block_height
        list_of_canals_to_add = [add_dam]
        
        while len(list_of_canals_to_add) > 0:
            list_of_canals_to_add = list(list_of_canals_to_add) # try to avoid numpyfication
            add_can = list_of_canals_to_add[0]
            
            if wt[add_can] < add_height: # condition for update
                wt[add_can] = add_height
                canals_prop_to = CNM[add_can].nonzero()[1].tolist() # look for propagation in matrix
                list_of_canals_to_add = list_of_canals_to_add + canals_prop_to # append canals to propagate to. If none, it appends nothing.
            
            list_of_canals_to_add = list_of_canals_to_add[1:] # remove add_can from list
            
        return wt
    
    wt = copy.deepcopy(originalWT) # Always start with original wt.
    
#    if type(dams_to_add) != list:
#        print("dams_to_add needs to be list type")
    
    for add_can in dams_to_add:
        wt = addDam(wt, srfc, block_height, add_can, CNM)
    
    return wt


def from_raster_pos_to_LatLong(positions_in_canal_network, c_to_r_list, can_network_raster_fn):
    
    import rasterio
    rows = [c_to_r_list[pos][0] for pos in positions_in_canal_network]
    cols = [c_to_r_list[pos][1] for pos in positions_in_canal_network]
    src = rasterio.open(can_network_raster_fn) # read metadata from input raster to copy it and have the same projection etc
    
    lats, longs = rasterio.transform.xy(transform=src.profile['transform'], rows=rows, cols=cols)
    
    latlongs = [(lats[i], longs[i]) for i in range(0, len(lats))]
    
    return latlongs
    
    
def map_lc_number_to_lc_coef_in_co2_emission_formula(lc_raster):
    # Coefficients taken from file 21092020_Formula for WTD conversion.xlsx
    # created by Imam and uploaded to Asana
    
    lc_multiplicative_coef_co2 = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_co2 = np.zeros(shape=lc_raster.shape)
    
    lc_multiplicative_coef_co2[(lc_raster==2002) | (lc_raster==20041) | (lc_raster==20051)] = -98 # Degraded forests
    lc_multiplicative_coef_co2[lc_raster==2005] = -98 # Primary forests
    lc_multiplicative_coef_co2[lc_raster==2006] = -69 # Acacia plantation
    lc_multiplicative_coef_co2[(lc_raster==2007) | (lc_raster==20071) | (lc_raster==20091) |
                (lc_raster==20092) | (lc_raster==20093) | (lc_raster==2012) |
                (lc_raster==20121) | (lc_raster==2014) | (lc_raster==20141) |
                (lc_raster==50011)] = -84 # Deforested peatlands
    lc_multiplicative_coef_co2[lc_raster==2010] = -77.07 # Other Plantations
    
    lc_additive_coef_co2[(lc_raster==2002) | (lc_raster==20041) | (lc_raster==20051)] = 0 # Degraded forests
    lc_additive_coef_co2[lc_raster==2005] = 0 # Primary forests
    lc_additive_coef_co2[lc_raster==2006] = 21 # Acacia plantation
    lc_additive_coef_co2[(lc_raster==2007) | (lc_raster==20071) | (lc_raster==20091) |
                (lc_raster==20092) | (lc_raster==20093) | (lc_raster==2012) |
                (lc_raster==20121) | (lc_raster==2014) | (lc_raster==20141) |
                (lc_raster==50011)] = 9 # Deforested peatlands
    lc_additive_coef_co2[lc_raster==2010] = 19.8 # Other Plantations

    return lc_multiplicative_coef_co2, lc_additive_coef_co2

def map_lc_number_to_lc_coef_in_subsidence_formula(lc_raster):
    # Coefficients taken from file 21092020_Formula for WTD conversion.xlsx
    # created by Imam and uploaded to Asana
    
    lc_multiplicative_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_subsi = np.zeros(shape=lc_raster.shape)
    
    lc_multiplicative_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_multiplicative_coef_subsi[(lc_raster==2002) | (lc_raster==20041) | (lc_raster==20051)] = -6.54 # Degraded forests
    lc_multiplicative_coef_subsi[lc_raster==2005] = -7.06 # Primary forests
    lc_multiplicative_coef_subsi[lc_raster==2006] = -4.98 # Acacia plantation
    lc_multiplicative_coef_subsi[(lc_raster==2007) | (lc_raster==20071) | (lc_raster==20091) |
                (lc_raster==20092) | (lc_raster==20093) | (lc_raster==2012) |
                (lc_raster==20121) | (lc_raster==2014) | (lc_raster==20141) |
                (lc_raster==50011)] = -5.98 # Deforested peatlands
    lc_multiplicative_coef_subsi[lc_raster==2010] = -4.98 # Other Plantations, same as Acacia
    
    lc_additive_coef_subsi = np.zeros(shape=lc_raster.shape)
    lc_additive_coef_subsi[(lc_raster==2002) | (lc_raster==20041) | (lc_raster==20051)] = 0.35 # Degraded forests
    lc_additive_coef_subsi[lc_raster==2005] = 0 # Primary forests
    lc_additive_coef_subsi[lc_raster==2006] = 1.5 # Acacia plantation
    lc_additive_coef_subsi[(lc_raster==2007) | (lc_raster==20071) | (lc_raster==20091) |
                (lc_raster==20092) | (lc_raster==20093) | (lc_raster==2012) |
                (lc_raster==20121) | (lc_raster==2014) | (lc_raster==20141) |
                (lc_raster==50011)] = 0.69 # Deforested peatlands
    lc_additive_coef_subsi[lc_raster==2010] = 1.5 # Other Plantations, same as Acacia
    
    
    return lc_multiplicative_coef_subsi, lc_additive_coef_subsi
    
def compute_co2_from_WTD(wtd, lc_multiplicative_coef_co2, lc_additive_coef_co2):
    wtd_hooijer_capped = wtd[:]
    wtd_carlson_capped = wtd[:]
    
    # WTD are capped because wtd for the co2 model in papers do not range further than -1.25
    # papers: hooijer, carlson, evans. From Imam's formula as discussed in Asana
    wtd_hooijer_capped[wtd < -1.25] = -1.25
    wtd_carlson_capped[wtd < -1.1] = -1.1
    
    co2_hooijer = lc_additive_coef_co2 + lc_multiplicative_coef_co2 * wtd_hooijer_capped #tCO2eq/ha/yr
    co2_carlson = lc_additive_coef_co2 + lc_multiplicative_coef_co2 * wtd_carlson_capped
    
    co2 = co2_hooijer
    carlson_mask = np.where(lc_multiplicative_coef_co2==-77.07)
    co2[carlson_mask] = co2_carlson[carlson_mask]
    
    return co2

def compute_subsi_from_WTD(wtd, lc_multiplicative_coef_subsi, lc_additive_coef_subsi):
    wtd_hooijer_capped = wtd[:]
    wtd_evans_capped = wtd[:]
    
    # WTD are capped because wtd for the co2 model in papers do not range further than -1.25
    # papers: hooijer, carlson, evans. From Imam's formula as discussed in Asana
    wtd_hooijer_capped[wtd < -1.25] = -1.25
    wtd_evans_capped[wtd < -1.2] = -1.2
    
    subsi_hooijer = lc_additive_coef_subsi + lc_multiplicative_coef_subsi * wtd_hooijer_capped #tCO2eq/ha/yr
    subsi_evans = lc_additive_coef_subsi + lc_multiplicative_coef_subsi * wtd_evans_capped
    
    subsi = subsi_hooijer
    evans_mask = np.where(lc_multiplicative_coef_subsi==-6.54)
    subsi[evans_mask] = subsi_evans[evans_mask]
    
    return subsi
    
    
    subsi = lc_additive_coef_subsi + lc_multiplicative_coef_subsi * wtd # cm/yr
    subsi = subsi/100 #m/yr
    
    return subsi

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    