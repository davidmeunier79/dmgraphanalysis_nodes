# -*- coding: utf-8 -*-
"""
Support function for net handling
"""

import time

import numpy as np
import scipy.sparse as sp 


def get_modularity_value_from_lol_file(modularity_file):
    
    with open(modularity_file,'r') as f:
        
        for line in f.readlines():
        
            split_line = line.strip().split(' ')
            
            print split_line
            
            if split_line[0] == 'Q':
            
                print "Found modularity value line"
                
                return split_line[2] 
                
                
        print "Unable to find modularity line in file, returning -1"
        
        return -1.0
        
################################################################# Node roles 

def return_all_Z_com_degree(community_vect,dense_mat):
    
    degree_vect = np.sum(dense_mat != 0,axis = 1)
    
    print "All degree vect"
    
    print degree_vect
    print degree_vect.shape
    
    community_indexes = np.unique(community_vect)
    
    #print community_indexes
    
    all_Z_com_degree = np.zeros(shape = (community_vect.shape[0]))
    
    for com_index in community_indexes:
    
        print np.where(com_index == community_vect)
        
        com_degree = degree_vect[com_index == community_vect]
        
        
        print "Commmunity degree vect"
        print com_degree
        print com_degree.shape
        
        if com_degree.shape[0] > 1:
            
            std_com_degree = np.std(com_degree)
            
            print std_com_degree
            
            mean_com_degree = np.mean(com_degree)
            
            print mean_com_degree
            Z_com_degree = (com_degree - mean_com_degree) / std_com_degree
            
            print Z_com_degree
            
            
            all_Z_com_degree[com_index == community_vect] = Z_com_degree
            
        else:
            all_Z_com_degree[com_index == community_vect] = 0
            
    return all_Z_com_degree
    
def return_all_participation_coeff(community_vect,dense_mat):

    degree_vect = np.array(np.sum(dense_mat != 0,axis = 1),dtype = 'float')
    
    print degree_vect
    print degree_vect.shape
    
    
    community_indexes = np.unique(community_vect)
    
    print community_indexes
    
    all_participation_coeff = np.ones(shape = (community_vect.shape[0]),dtype = 'float')
    
    for com_index in community_indexes:
    
        print np.where(com_index == community_vect)
        
        nod_index = (com_index == community_vect)
        
        print np.sum(nod_index,axis = 0)
        
        com_matrix = dense_mat[:,nod_index]
        
        degree_com_vect = np.sum(com_matrix,axis = 1,dtype = 'float')
        
        print degree_com_vect.shape

        rel_com_degree = np.square(degree_com_vect/degree_vect)
        
        print rel_com_degree
        
        all_participation_coeff = all_participation_coeff - rel_com_degree
        
    print all_participation_coeff
        
    return all_participation_coeff

def return_amaral_roles(all_Z_com_degree ,all_participation_coeff):
    
    if (all_Z_com_degree.shape[0] != all_participation_coeff.shape[0]):
        print "Warning, all_Z_com_degree %d should have same length as all_participation_coeff %d "%(all_Z_com_degree.shape[0],all_participation_coeff.shape[0])
        return 0
        
        
    nod_roles = np.zeros(shape = (all_Z_com_degree.shape[0],2),dtype = 'int')
    
    ### hubs are at 2,non-hubs are at 1
    hubs = all_Z_com_degree > 2.5
    non_hubs =  all_Z_com_degree <= 2.5
    
    nod_roles[hubs,0] = 2
    nod_roles[non_hubs,0] = 1
    
    ### for non-hubs
    #ultraperipheral nodes
    ultraperi_non_hubs = np.logical_and(all_participation_coeff < 0.05,non_hubs == True)
    
    print np.sum(ultraperi_non_hubs,axis = 0)
    nod_roles[ultraperi_non_hubs,1] = 1
    
    #ultraperipheral nodes
    peri_non_hubs = np.logical_and(np.logical_and(0.05 <= all_participation_coeff,all_participation_coeff < 0.62),non_hubs == True)
    
    print np.sum(peri_non_hubs,axis = 0)
    nod_roles[peri_non_hubs,1] = 2
    
    #non-hub connectors
    non_hub_connectors = np.logical_and(np.logical_and(0.62 <= all_participation_coeff,all_participation_coeff < 0.8),non_hubs == True)
    
    print np.sum(non_hub_connectors,axis = 0)
    nod_roles[non_hub_connectors,1] = 3
    
    
    #kinless non-hubs
    kin_less_non_hubs = np.logical_and(0.8 <= all_participation_coeff,non_hubs == True)
    
    print np.sum(kin_less_non_hubs,axis = 0)
    nod_roles[kin_less_non_hubs,1] = 4
    
    ### for hubs
    #provincial hubs
    prov_hubs = np.logical_and(all_participation_coeff < 0.3,hubs == True)
    
    print np.sum(prov_hubs,axis = 0)
    nod_roles[prov_hubs,1] = 5
    
    
    #hub connectors
    hub_connectors = np.logical_and(np.logical_and(0.3 <= all_participation_coeff,all_participation_coeff < 0.75),hubs == True)
    
    print np.sum(hub_connectors,axis = 0)
    nod_roles[hub_connectors,1] = 6
    
    
    #kinless hubs
    kin_less_hubs = np.logical_and(0.75 <= all_participation_coeff,hubs == True)
    
    print np.sum(kin_less_hubs,axis = 0)
    nod_roles[kin_less_hubs,1] = 7
    
    print nod_roles
    
    return nod_roles
    
def return_4roles(all_Z_com_degree ,all_participation_coeff):
    
    if (all_Z_com_degree.shape[0] != all_participation_coeff.shape[0]):
        print "Warning, all_Z_com_degree %d should have same length as all_participation_coeff %d "%(all_Z_com_degree.shape[0],all_participation_coeff.shape[0])
        return 0
        
        
    nod_roles = np.zeros(shape = (all_Z_com_degree.shape[0],2),dtype = 'int')
    
    ### hubs are at 2,non-hubs are at 1
    hubs = all_Z_com_degree > 1.0
    non_hubs =  all_Z_com_degree <= 1.0
    
    print np.sum(hubs,axis = 0),np.sum(non_hubs,axis = 0)
    
    nod_roles[hubs,0] = 2
    nod_roles[non_hubs,0] = 1
    
    #provincial nodes
    provincial_nodes = all_participation_coeff < 0.3
    
    nod_roles[provincial_nodes,1] = 1
    
    #connector nodes
    connector_nodes = 0.3 <= all_participation_coeff
    
    print np.sum(provincial_nodes,axis = 0),np.sum(connector_nodes,axis = 0)
    nod_roles[connector_nodes,1] = 2
    
    #print nod_roles
    
    return nod_roles
    
def compute_roles(community_vect,sparse_mat, role_type = "Amaral_roles"):

    import numpy as np
    
    dense_mat = sparse_mat.todense()
    
    print dense_mat
    
    undir_dense_mat = dense_mat + np.transpose(dense_mat)
    
    bin_dense_mat = np.array(undir_dense_mat != 0,dtype = int)
    
    print bin_dense_mat
    
    ##################################### within community Z-degree #########################

    all_Z_com_degree = return_all_Z_com_degree(community_vect,bin_dense_mat)
    
    print all_Z_com_degree
    
    ##################################### participation_coeff ###############################
    
    all_participation_coeff = return_all_participation_coeff(community_vect,bin_dense_mat)
    
    print all_participation_coeff
    
    if role_type == "Amaral_roles":
        
        node_roles = return_amaral_roles(all_Z_com_degree ,all_participation_coeff)
    
    elif role_type == "4roles":
    
        node_roles = return_4roles(all_Z_com_degree ,all_participation_coeff)
    
    
    
    return node_roles,all_Z_com_degree,all_participation_coeff
