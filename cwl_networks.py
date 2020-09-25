# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:31:30 2020

@author: 03125327
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import preprocess_data

#%%
# Canal network and DEM data
true_data = False
if not true_data:
    CNM = np.array([[0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0],
                    [0,0,0,0,1],
                    [0,0,0,0,0]])
    dem_nodes = np.array([10, 5, 4, 3, 2]) # m.a.s.l.


else:
    filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)
    
    dem_rst_fn = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
    can_rst_fn = Path(filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
    # TODO: use peat depth as depth of canals??
    peat_depth_rst_fn = Path(filenames_df[filenames_df.Content == 'peat_depth_raster'].Path.values[0])

    # Choose smaller study area
    STUDY_AREA = (0,-1), (0,-1) # E.g., a study area of (0,-1), (0,-1) is the whole domain
    
    if 'CNM' and 'dem_nodes' and 'c_to_r_list' not in globals():
        CNM, cr, c_to_r_list = preprocess_data.gen_can_matrix_and_raster_from_raster(sa=STUDY_AREA, can_rst_fn=can_rst_fn, dem_rst_fn=dem_rst_fn)
        _, wtd_old , dem, peat_type_arr, peat_depth_arr = preprocess_data.read_preprocess_rasters(STUDY_AREA, dem_rst_fn, can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn)
        dem_nodes = [dem[loc] for loc in c_to_r_list]
        dem_nodes[0] = 3.0 # something strange happens with this node
        dem_nodes = np.array(dem_nodes)
        


n_edges = np.sum(CNM)
n_nodes = CNM.shape[0]
DIST_BETWEEN_NODES = 100. # m

#%%
# Hydrology-relevant data: BC, IC, rainfall, etc.

def infer_BC_nodes(adj_matrix):
    """
    Infer what nodes are Neumann and Dirichlet from adjacency matrix.
    Last nodes of canals (identified by having no outgoing edges) are set to Diri BC
    First nodes of canals (No ingoing edges) are set to Neumann

    Parameters
    ----------
    adj_matrix : numpy array
        Adjacency matrix of the canal network graph

    Returns
    -------
    diri_bc_bool : boolean numpy array
        True where nodes have Dirichlet BC
    neumann_bc_bool : boolean numpy array
        True where nodes have Neumann BC

    """
    # Infer neumann and Diri nodes from adj matrix
    neumann_bc_bool = np.sum(CNM, axis=0) == 0 # Boundary values below are conditional on this boolean mask
    diri_bc_bool = np.sum(CNM, axis=1) == 0 
    # in case the summing over the sparse matrix changes the numpy array shape
    neumann_bc_bool = np.ravel(neumann_bc_bool) 
    diri_bc_bool = np.ravel(diri_bc_bool)
   
    return diri_bc_bool, neumann_bc_bool

# BC    
# For now, same BC across the domain, but could be spatio-temporaly varying
diri_bc_bool, neumann_bc_bool = infer_BC_nodes(CNM)

DIRI_BC = 1.
NEUMANN_BC = 0.
diri_bc_nodes = DIRI_BC * diri_bc_bool
neumann_bc_nodes = NEUMANN_BC * neumann_bc_bool

# P - ET
SOURCE = 0.0 # P - ET m/ dt units
source = np.ones(shape=n_nodes) * SOURCE # homogeneous rainfall & ET

# ini cond
CWL_INI = 0. # metres below DEM surface. Negative downwards
cwl_ini = np.ones(shape=n_nodes) * CWL_INI
h_ini = dem_nodes + cwl_ini

#%%
# create graph
g = nx.DiGraph(incoming_graph_data=CNM.T) # transposed for dynamics!
g_un = nx.Graph(g) # symmetric matrix, undirected graph. Useful for dynamics 

nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_old')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_new')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(dem_nodes)}, name='ele')   
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(diri_bc_nodes)}, name='diri_bc')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(diri_bc_bool)}, name='diri_bool')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(neumann_bc_nodes)}, name='neumann_bc')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(neumann_bc_bool)}, name='neumann_bool')
nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(source)}, name='source')      

# The following is valid if conductivity and storage vary from place to place.
# conductivity = np.array([1]*n_edges)
# storage = np.array([1]*n_nodes)
# if len(conductivity) != n_edges:
#     raise ValueError("""conductivity is an edge property, there should as many
#                      edges as there are conductivities""")
# if len(storage) != n_nodes:
#     raise ValueError("""storage is a node property, there should as many
#                      nodes as there are storages""")
# nx.set_node_attributes(g, values={i: value for i, value in enumerate(storage)}, name='S')
# nx.set_edge_attributes(g, values={edge: conductivity[n] for n, edge in enumerate(g.edges)}, name='K')

# Otherwise, they can be constants
K = 1.
S = 1.
              
#%%
# Dynamics

# types can be 'advection' for simple gradient transport or 'diffuson' for 2nd order derivative 
EQ_TYPE = 'advection' 

if EQ_TYPE=='diffusion':
    # grad(h) is a new edge variable that allows to compute d^2h/dx^2 = d grad(h)/dx
    nx.set_node_attributes(G=g, values=0, name='grad')


dt = 1.
dx = DIST_BETWEEN_NODES
niter = 100
h = h_ini

plt.figure()

# TODO: Only mock dynamics here. Implement Saint Venants simplification
for t in range(niter):
    print(t)
    for n, node in g.nodes(data=True):
        if node['diri_bool']: # Diri BC
            continue
        else:
            h_incr = 0
            for i in g_un.neighbors(n): # all neighbors, not just in or outgoing
                neigh = g.nodes[i]
                if EQ_TYPE=='advection':
                    h_incr = h_incr + dt/dx * K/S * (neigh['h_old'] - node['h_old']) 
            
                # elif EQ_TYPE=='diffusion':
                    
                    
                    
                else:
                    raise ValueError('Type of equation not understood')
            if node['neumann_bool']: # Neumann BC
                h_incr = h_incr + dt * K/S * node['neumann_bc']
                
            node['h_new'] = node['h_old'] + h_incr + dt/S * node['source']
    
    # Print some stuff
    hs = [f"{node['h_old']:.2f}" for n, node in g.nodes(data=True)]
    hss = [node['h_old'] for n, node in g.nodes(data=True)]
    print(hs, f'sum = {sum(hss)}')
    plt.plot(range(n_nodes), hss, label=f'iter:{t}', color='grey', alpha=0.3)
    
    # update old to new
    for n, node in g.nodes(data=True):
        node['h_old'] = node['h_new']
    
    
#%%
# Vectorized. 
# # check validity of matrix approach & sign of gradient
# incoming_h =  np.matmul(np.diag(dem_nodes), A)
# outgoing_h =  np.matmul(A, np.diag(dem_nodes))
# gradients = (incoming_h - outgoing_h) / dist_between_nodes


