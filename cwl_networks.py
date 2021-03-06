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
import scipy.sparse
import warnings

import preprocess_data

plotOpt = True

#%%
# Canal network and DEM data
true_data = False
if not true_data:
    cnm = np.array([[0,0,0,0,0],
                    [1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0]])
    dem_nodes = np.array([10, 5, 4, 3, 2]) # m.a.s.l.
    
    cnm_sim = cnm + cnm.T

    CNM = scipy.sparse.csr_matrix(cnm) # In order to use the same sparse type as the big ass true adjacency matrix

else:
    filenames_df = pd.read_excel('file_pointers.xlsx', header=2, dtype=str)
    
    dem_rst_fn = Path(filenames_df[filenames_df.Content == 'DEM'].Path.values[0])
    can_rst_fn = Path(filenames_df[filenames_df.Content == 'canal_raster'].Path.values[0])
    # TODO: use peat depth as depth of canals??
    peat_depth_rst_fn = Path(filenames_df[filenames_df.Content == 'peat_depth_raster'].Path.values[0])

    # Choose smaller study area
    STUDY_AREA = (0,-1), (0,-1) # E.g., a study area of (0,-1), (0,-1) is the whole domain
    
    can_arr, wtd_old , dem, _, peat_depth_arr, _, _ = preprocess_data.read_preprocess_rasters(STUDY_AREA, dem_rst_fn, can_rst_fn, dem_rst_fn, peat_depth_rst_fn, peat_depth_rst_fn, dem_rst_fn, dem_rst_fn)  
    labelled_canals = preprocess_data.label_canal_pixels(can_arr, dem)
    CNM, c_to_r_list = preprocess_data.gen_can_matrix_and_label_map(labelled_canals, dem)
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
    diri_bc_bool = np.sum(CNM, axis=0) == 0 # Boundary values below are conditional on this boolean mask
    neumann_bc_bool = np.sum(CNM, axis=1) == 0 
    # in case the summing over the sparse matrix changes the numpy array shape
    diri_bc_bool = np.ravel(diri_bc_bool) 
    neumann_bc_bool = np.ravel(neumann_bc_bool)
   
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

def initialize_graph_values(g, h_ini, dem_nodes, diri_bc_nodes, diri_bc_bool, neumann_bc_nodes, neumann_bc_bool, source):
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_old')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(h_ini)}, name='h_new')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(dem_nodes)}, name='ele')   
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(diri_bc_nodes)}, name='diri_bc')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(diri_bc_bool)}, name='diri_bool')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(neumann_bc_nodes)}, name='neumann_bc')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(neumann_bc_bool)}, name='neumann_bool')
    nx.set_node_attributes(G=g, values={i: value for i, value in enumerate(source)}, name='source')
    
    return 0

g_un = nx.Graph(g) # symmetric matrix, undirected graph. Useful for dynamics    

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

# Other params
dt = 1.
dx = DIST_BETWEEN_NODES
niter = 100
              
#%%
# Dynamics with NetworkX
def advection_netx_single_step(g, dt, dx, b):
    """
    Performs advection on graph and updates properties in place
    Diffusion not yet implemented!
    - b is advection constant, i.e., du/dt = bdu/dx
    - g is undirected
    """  
    for n, node in g.nodes(data=True):
        if node['diri_bool']: # Diri BC
            continue # Assuming here that initial condition already satisfies Diri BC
        else:
            h_incr = 0
            for i in g.neighbors(n): # all neighbors, not just in or outgoing
                neigh = g.nodes[i]
                h_incr = h_incr + b/dx * (neigh['h_old'] - node['h_old'])
                
            if node['neumann_bool']: # Neumann BC
                h_incr = h_incr + node['neumann_bc']/dx
                
            # Synchronous update
            node['h_new'] = node['h_old'] + dt*(h_incr + node['source'])

    return 0

initialize_graph_values(g_un, h_ini, dem_nodes, diri_bc_nodes, diri_bc_bool, neumann_bc_nodes, neumann_bc_bool, source)

h_adv_nx = [[] for i in range(niter)]
for t in range(niter):
    
    h_adv_nx[t] = [node['h_new'] for _,node in g_un.nodes(data=True)]
    
    advection_netx_single_step(g_un, dt, dx, 1)
    
    for n, node in g_un.nodes(data=True):
        node['h_old'] = node['h_new']

plt.figure()
for t in range(niter):
    plt.plot(h_adv_nx[t], color='blue', alpha=0.3)


#%%
# Vectorally
def compute_laplacian_from_adjacency(adj_matrix):
    if np.any(adj_matrix != adj_matrix.T):
        raise ValueError('the matrix must be symmetric, i.e., must be the adj matrix of an undirected graph')
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix
    
    return laplacian

def L_advection(directed_adj_matrix):
    """
    Returns the 'modified Laplacian', i.e., the advection operator
    """
    D_out = np.diag(np.sum(directed_adj_matrix, axis=0))
    return D_out - directed_adj_matrix

def advection_diffusion_operator(dx, L, L_adv, a, b):
    return -a/dx**2*L + b/dx*L_adv

def set_Neumann_BC(L_mix, L, neumann_bc_bool):
    
    warnings.warn('Neumann BC not checked yet!')
    
    no_flux_nodes = np.where(neumann_bc_bool)[0]
    for nrow in no_flux_nodes: # No flux BC
        L_mix[nrow] = L[nrow]
    
    return L_mix

def set_Diri_BC(h, h_old, diri_bc_bool):
    h = np.where(diri_bc_bool, h_old, h) # Diri BC conditions    
    return h

def forward_Euler_adv_diff_single_step(h, h_old, dt, dx, a, b, L, L_adv, source, diri_bc_bool, neumann_bc_bool):
    
    L_mix = advection_diffusion_operator(dx, L, L_adv, a, b)
    
    L_mix = set_Neumann_BC(L_mix, L, neumann_bc_bool)
    
    h = h - dt * L_mix @ h + dt*source
    
    h = set_Diri_BC(h, h_old, diri_bc_bool)
    
    return h

def backwards_Euler(h, h_old, dt, dx, a, b, L, L_adv, source, diri_bc_bool, neumann_bc_bool):
    
    L_mix = advection_diffusion_operator(dx, L, L_adv, a, b)
    
    L_mix = set_Neumann_BC(L_mix, L, neumann_bc_bool)
    
    P = np.eye(N=L.shape[0]) - dt*L_mix
    P_inv = np.linalg.inv(P)
    
    h = P_inv @ (h + dt*source)
    
    h = set_Diri_BC(h, h_old, diri_bc_bool)
    
    return h

def advection_diffusion_vectorial(dt, dx, a, b, niter, h_ini, A, source):
    """
    Advection and diffusion terms.
    eq: dm/dt = am'' + bm' + source

    """
    h = h_ini[:] # Ini cond
    h_old = h_ini[:]

    L = compute_laplacian_from_adjacency(cnm_sim)
    L_adv = L_advection(cnm)
    
    for t in range(niter):
       h = forward_Euler_adv_diff_single_step(h, h_old, dt, dx, a, b, L, L_adv, source, diri_bc_bool)
    return(h)

# Plot solutions over time
if plotOpt:
    niter = 10000
    dt = 100/niter
    dx = 1
    
    h_adv = h_ini[:]
    h_dif = h_ini[:]
    h_advdif = h_ini[:]
    L = compute_laplacian_from_adjacency(cnm_sim)
    L_adv = L_advection(cnm)
    
    plt.figure()
    for t in range(niter):
        if t % (int(niter/10))==0:
            plt.plot(h_adv, color='blue', alpha=0.5, label='advection')
            plt.plot(h_dif, color='orange', alpha=0.5, label='diffusion')
            plt.plot(h_advdif, color='green', alpha=0.5, label='adv + diff')
        h_adv = forward_Euler_adv_diff_single_step(h_adv, h_ini, dt, dx, 0, 1, L, L_adv, source, diri_bc_bool, neumann_bc_bool)
        h_dif = forward_Euler_adv_diff_single_step(h_dif, h_ini, dt, dx, 1, 0, L, L_adv, source, diri_bc_bool, neumann_bc_bool)
        h_advdif = forward_Euler_adv_diff_single_step(h_advdif, h_ini, dt, dx, 1, 1, L, L_adv, source, diri_bc_bool, neumann_bc_bool)
        
    plt.legend()
    plt.show()
        
        
#%%
# Check with standard 1d finite differences. Comparison only valid when network = 1d linear mesh
    
h_ini = dem_nodes + cwl_ini

def advection_fd(dt, dx, niter, h_ini):
    h = h_ini[:]
    h_old = h_ini[:]
    N = len(h)
    
    for t in range(niter):
        for i in range(N):
            if i==0: # Neumann BC   
                h[0] = h_old[0] + dt/dx *0.5* (-h_old[0] + h_old[1]) + dt*source[0]
            elif i==N-1: # Diri BC
                h[-1] = h_old[-1]
            else:
                # backward difference
                h[i] = h_old[i] + dt/dx * (h_old[i] - h_old[i-1])
                
        h_old = h
        
    return h

def diffusion_fd(dt, dx, niter, h_ini):
    h = h_ini[:]
    h_old = h_ini[:]
    N = len(h)
    
    for t in range(niter):
        for i in range(N):
            if i==0: # No flux Neumann BC   
                h[0] = h_old[0] + dt/dx * 0.5* (-h_old[0] + h_old[1]) + dt*source[0]
            elif i==N-1: # Diri BC
                h[-1] = h_old[-1]
            else:
                h[i] = h_old[i] + dt/dx**2 *  (h_old[i-1] - 2*h_old[i] + h_old[i+1]) + dt*source[i]
                
        h_old = h
        
    return h
                
            
#%%
# Comparison graph and finite differences approach (only valid for 1d, linear mesh)
h_ini = dem_nodes + cwl_ini

sol_fdadv = advection_fd(dt, dx, niter, dem_nodes + cwl_ini)
sol_fddif = diffusion_fd(dt, dx, niter, dem_nodes + cwl_ini)        

sol_adv = advection_diffusion_vectorial(dt, dx, 0, 1, niter, dem_nodes + cwl_ini, cnm_sim, source)
sol_dif = advection_diffusion_vectorial(dt, dx, 1, 0, niter, dem_nodes + cwl_ini, cnm_sim, source)
                     
            
            
            
            
            
            
            
            
            
            
            
            
            
            