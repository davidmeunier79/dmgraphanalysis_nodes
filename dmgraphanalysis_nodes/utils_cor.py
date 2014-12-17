# -*- coding: utf-8 -*-
"""
Support function for run_wei_sig_correl_mat.py, run_wei_sig_modularity.py and run_gather_partitions
"""

import sys, os


from scipy import stats
import numpy as np

#import pandas as pd

import itertools as it

#import scipy.io
#import scipy.spatial.distance as dist
##import math

import scipy.signal as filt

#import scipy.sparse as sp 
#import scipy.cluster.hierarchy as hie


import time

#from collections import Counter

#from nipype.utils.filemanip import split_filename as split_f
    
#from dipy.align.aniso2iso import resample

#from sets import Set

from utils_dtype_coord import *

    
def return_conf_cor_mat(ts_mat,regressor_vect,conf_interval_prob):
    
    t1 = time.time()
    
    if ts_mat.shape[0] != len(regressor_vect):
        "Warning, incompatible regressor length {} {}".format(ts_mat.shape[0], len(regressor_vect))
        return

    
    keep = regressor_vect > 0.0
    w = regressor_vect[keep]
    ts_mat = ts_mat[keep,:]
    
    ### confidence interval for variance computation
    norm = stats.norm.ppf(1-conf_interval_prob/2)
    #deg_freedom = w.sum()-3
    deg_freedom = w.sum()/w.max()-3
    #deg_freedom = w.shape[0]-3
    
    print deg_freedom
    
    print norm,norm/np.sqrt(deg_freedom)
    
    print regressor_vect.shape[0],w.shape[0],w.sum(),w.sum()/w.max()
    s, n = ts_mat.shape
    
    Z_cor_mat = np.zeros((n,n),dtype = float)
    Z_conf_cor_mat = np.zeros((n,n),dtype = float)
    
    cor_mat = np.zeros((n,n),dtype = float)
    conf_cor_mat = np.zeros((n,n),dtype = float)
    
    ts_mat2 = ts_mat*np.sqrt(w)[:,np.newaxis]
    
    for i,j in it.combinations(range(n), 2):
    
        s1 = ts_mat2[:,i]
        s2 = ts_mat2[:,j]
        
        cor_mat[i,j] = (s1*s2).sum()/np.sqrt((s1*s1).sum() *(s2*s2).sum())
        Z_cor_mat[i,j] = np.arctanh(cor_mat[i,j])
        
        Z_conf_cor_mat[i,j] = norm/np.sqrt(deg_freedom)
        
        if cor_mat[i,j] > 0:
            conf_cor_mat[i,j] = cor_mat[i,j] - np.tanh(Z_cor_mat[i,j] - norm/np.sqrt(deg_freedom))
        else:
            conf_cor_mat[i,j] = - cor_mat[i,j] + np.tanh(Z_cor_mat[i,j] + norm/np.sqrt(deg_freedom))
            
        
        #print i,j,cor_mat[i,j],conf_cor_mat[i,j]
        
    t2 = time.time()
    
    print "Weighted correlation computation took " + str(t2-t1) + "s"
    
    return cor_mat,Z_cor_mat,conf_cor_mat,Z_conf_cor_mat
    
    
###################################### coclassification matrix 

def return_coclass_mat(community_vect,corres_coords,gm_mask_coords):

    print corres_coords.shape[0],community_vect.shape[0]
    
    if (corres_coords.shape[0] != community_vect.shape[0]):
        print "warning, length of corres_coords and community_vect are imcompatible {} {}".format(corres_coords.shape[0],community_vect.shape[0])
    
    where_in_gm = where_in_coords(corres_coords,gm_mask_coords)
    
    print where_in_gm
    
    print np.min(where_in_gm),np.max(where_in_gm),where_in_gm.shape
    
    if (where_in_gm.shape[0] != community_vect.shape[0]):
        print "warning, length of where_in_gm and community_vect are imcompatible {} {}".format(where_in_gm.shape[0],community_vect.shape[0])
    
    coclass_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        
    possible_edge_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
    
    for i,j in it.combinations(range(where_in_gm.shape[0]),2):
    
        coclass_mat[where_in_gm[i],where_in_gm[j]] = np.int(community_vect[i] == community_vect[j])
        coclass_mat[where_in_gm[j],where_in_gm[i]] = np.int(community_vect[i] == community_vect[j])
        
        possible_edge_mat[where_in_gm[i],where_in_gm[j]] = 1
        possible_edge_mat[where_in_gm[j],where_in_gm[i]] = 1
        
    return coclass_mat,possible_edge_mat
    
    
    
#def return_hierachical_order(mat,order_method):

    ##vect = coclass_mat[np.triu_indices(coclass_mat.shape[0], k=1)]
    
    ##print vect.shape
    
    ##linkageMatrix = hier.linkage(coclass_mat,method='centroid')
    #linkageMatrix = hie.linkage(mat,method = order_method)
    
    ##print linkageMatrix.shape
    
    #dendro = hie.dendrogram(linkageMatrix,get_leaves = True)

    
    
    ##print dendro
    
    ###get the order of rows according to the dendrogram 
    #leaves = dendro['leaves'] 
    
    ##print leaves
    
    #mat = mat[leaves,: ]
    
    #reorder_mat = mat[:, leaves]
    
    ##print reorder_mat.shape
    
    #return reorder_mat,leaves
    
def return_corres_correl_mat(Z_cor_mat,coords,gm_mask_coords):
    
    #print coords
    #print gm_mask_coords
    
    where_in_gm = where_in_coords(coords,gm_mask_coords)
    
    #print where_in_gm
    
    print np.min(where_in_gm),np.max(where_in_gm),where_in_gm.shape
    
    
    corres_correl_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = float)
    possible_edge_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
    
    for i,j in it.combinations(range(len(where_in_gm)),2):
    
        corres_correl_mat[where_in_gm[i],where_in_gm[j]] = Z_cor_mat[i,j]
        corres_correl_mat[where_in_gm[j],where_in_gm[i]] = Z_cor_mat[i,j]
        
        possible_edge_mat[where_in_gm[i],where_in_gm[j]] = 1
        possible_edge_mat[where_in_gm[j],where_in_gm[i]] = 1
        
    return corres_correl_mat,possible_edge_mat
