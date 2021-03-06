# -*- coding: utf-8 -*-
"""
Support function for run_wei_sig_correl_mat.py, run_wei_sig_modularity.py and run_gather_partitions
"""

import sys, os


from scipy import stats
import numpy as np

import pandas as pd
    
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

#def return_regressor(spm_mat_file,regressor_name):

    ###Reading spm.mat for regressors extraction:
    #d = scipy.io.loadmat(spm_mat_file)
    
    ###Choosing the column according to the regressor name
    #_,col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == u'Sn(1) ' + regressor_name)
    
    ### reformating matrix (1,len) in vector (len)
    #regressor_vect = d['SPM']['xX'][0][0]['X'][0][0][:,col].reshape(-1)
    
    ### Keeping only the non-negative values
    ### In original (Dodel et al., 2005, Proc Roy Soc B) they used absolute values, thus retaining the small rebound of hemodynamic function
    ### part as positive values, but that does not make sense to me...
    #regressor_vect[regressor_vect < 0] = 0
    
    #return regressor_vect
    
########################## resample data #############################################################

#def resample_img(img_file,vox_dim):
    
    #print "loading img"
    
    #img = nib.load(img_file)

    #img_data = img.get_data()    
    #img_affine = img.get_affine()    
    #img_header = img.get_header()
    #img_cur_vox_dim = img_header.get_zooms()[:3]
    
    ##print img_data.shape    
    ##print img_affine    
    ##print img_cur_vox_dim
    #print "resampling img"
    
    #resampled_img_data,resampled_img_affine = resample(img_data,img_affine,img_cur_vox_dim,vox_dim)
    
    ##print resampled_img_data.shape    
    ##print resampled_img_affine
    
    #print "saving resampled img"
    
    #path,fname_img,ext_img = split_f(img_file)
    #resampled_img_file = os.path.join(path,"r" + fname_img + ext_img)
    #nib.save(nib.Nifti1Image(resampled_img_data,resampled_img_affine,img_header),resampled_img_file)
    
    #return resampled_img_file
    
    
    
      
########################## select data with original (binary) mask ###################################

#def select_data(data_img,data_mask):
    
    #img_shape = data_img.shape
    #mask_shape = data_mask.shape
    
    #non_null_mask = data_mask > min_non_null_mask_value
    
    #coords = np.column_stack(np.where(non_null_mask))
    
    #print coords
    ##print coords.shape
    
    ##print data_mask.size
    ##print np.sum(data_mask == 1)
    
    #if img_shape[:3] == mask_shape:
        #print "Image and mask are compatible"
        
        #masked_data_matrix = data_img[non_null_mask,:]
        #print masked_data_matrix.shape
        
        
        #non_zeros_indexes = np.sum(masked_data_matrix < min_BOLD_intensity,axis = 1) == 0
        #print non_zeros_indexes.shape
        
        #non_zeros_data_matrix = masked_data_matrix[non_zeros_indexes,:]
        ##print non_zeros_data_matrix.shape
        
        #non_zeros_coords = coords[non_zeros_indexes,:]
        ##print non_zeros_coords.shape
        
    #else:
        #print "Warning, Image and mask are incompatible"
        #print img_shape
        #print data_mask
        #return
    
    #return np.array(non_zeros_data_matrix),non_zeros_coords
    
#def mean_select_data(data_img,data_mask):
    
    #img_shape = data_img.shape
    #mask_shape = data_mask.shape
    
    #non_null_mask = data_mask > min_non_null_mask_value
    
    #coords = np.column_stack(np.where(non_null_mask))
    
    #print coords
    ##print coords.shape
    
    ##print data_mask.size
    ##print np.sum(data_mask == 1)
    
    #if img_shape[:3] == mask_shape:
        #print "Image and mask are compatible"
        
        #masked_data_matrix = data_img[non_null_mask,:]
        #print masked_data_matrix.shape
        
        #non_zeros_indexes = np.sum(masked_data_matrix < min_BOLD_intensity,axis = 1) == 0
        #print non_zeros_indexes.shape
        
        #non_zeros_data_matrix = masked_data_matrix[non_zeros_indexes,:]
        ##print non_zeros_data_matrix.shape
        
        #non_zeros_coords = coords[non_zeros_indexes,:]
        ##print non_zeros_coords.shape
        
        #mean_non_zeros_data_matrix = np.mean(non_zeros_data_matrix,axis = 0)
        
        #print mean_non_zeros_data_matrix.shape
    #else:
        #print "Warning, Image and mask are incompatible"
        #print img_shape
        #print data_mask
        #return
    
    #return mean_non_zeros_data_matrix
    
########################## select data with resampled or normalized mask ###################################


#def return_coord_non_null_mask(data_mask):
    
    #coords = np.column_stack(np.where(data_mask > min_non_null_mask_value))
    
    #return coords
    
def mean_select_mask_data(data_img,data_mask):
    
    img_shape = data_img.shape
    mask_shape = data_mask.shape
    
    #print coords
    #print coords.shape
    
    #print data_mask.size
    #print np.sum(data_mask == 1)
    
    if img_shape[:3] == mask_shape:
        print "Image and mask are compatible"
        
        masked_data_matrix = data_img[data_mask == 1,:]
        print masked_data_matrix.shape
        
        mean_mask_data_matrix = np.mean(masked_data_matrix,axis = 0)
        print mean_mask_data_matrix.shape
        
    else:
        print "Warning, Image and mask are incompatible"
        print img_shape
        print data_mask
        return
    
    return np.array(mean_mask_data_matrix)
    
def mean_select_indexed_mask_data(orig_ts,indexed_mask_rois_data,coord_rois,min_BOLD_intensity = 50):
        
        ### extrating ts by averaging the time series of all voxel with the same index
        sequence_roi_index = np.array(np.unique(indexed_mask_rois_data),dtype = int)
        
        if sequence_roi_index[0] == -1.0:
            sequence_roi_index = sequence_roi_index[1:]
        
        #print "sequence_roi_index:"
        #print sequence_roi_index
        
        if sequence_roi_index.shape[0] != coord_rois.shape[0]:
            print "Warning, indexes in template_ROI are incompatible with ROI coords"
            return
    
        mean_masked_ts = []
        subj_coord_rois = []
        
        for roi_index in sequence_roi_index:
        #for roi_index in sequence_roi_index[0:1]:
            
            #print "Roi index " + str(roi_index)
            
            index_roi_x,index_roi_y,index_roi_z = np.where(indexed_mask_rois_data == roi_index)
            
            print index_roi_x,index_roi_y,index_roi_z
            
            all_voxel_roi_ts = orig_ts[index_roi_x,index_roi_y,index_roi_z,:]
            
            print all_voxel_roi_ts.shape
            
            #print "all_voxel_roi_ts shape: "
            #print all_voxel_roi_ts.shape
            
            ### testing if at least 50% of the voxels in the ROIs have values always higher than min bold intensity
            if np.sum(np.sum(all_voxel_roi_ts > min_BOLD_intensity,axis = 1) == all_voxel_roi_ts.shape[1])/float(all_voxel_roi_ts.shape[0]) > 0.5:
                
                #print "Roi selected: " + str(roi_index)
                #print "coord_rois shape: "
                #print coord_rois.shape
                
                mean_roi_ts = np.mean(all_voxel_roi_ts,axis = 0)
            
                subj_coord_rois.append(coord_rois[roi_index,])
                mean_masked_ts.append(mean_roi_ts)
            else:
                print "ROI " + str(roi_index) + " was not selected"
                
                
            ### testing if mean_roi_ts if always higher than minimal BOLD intensity
            #if sum(mean_roi_ts > min_BOLD_intensity) == mean_roi_ts.shape[0]:
                
                ##print "Roi selected: " + str(roi_index)
                ##print "coord_rois shape: "
                ##print coord_rois.shape
                
                #subj_coord_rois.append(coord_rois[roi_index,])
                #mean_masked_ts.append(mean_roi_ts)
            #else:
                #print "ROI " + str(roi_index) + " was not selected"
                
            
        mean_masked_ts = np.array(mean_masked_ts,dtype = 'f')
        subj_coord_rois = np.array(subj_coord_rois,dtype = 'float')
        
        print mean_masked_ts.shape
        
        return mean_masked_ts,subj_coord_rois
        
    
#def mean_select_non_null_data(data_img,data_mask):
      
    #img_shape = data_img.shape
    #mask_shape = data_mask.shape
    
    #print img_shape
    #print mask_shape
    
    #coords = return_coord_non_null_mask(data_mask)
    
    ##print coords
    ##print coords.shape
    
    ##print data_mask.size
    ##print np.sum(data_mask == 1)
    
    #if img_shape[:3] == mask_shape:
        #print "Image and mask are compatible"
        
        #masked_data_matrix = data_img[data_mask > min_non_null_mask_value,:]
        ##print masked_data_matrix.shape
       
        
        #non_zeros_indexes = np.sum(masked_data_matrix < min_BOLD_intensity,axis = 1) == 0
        #print non_zeros_indexes.shape
        
        #non_zeros_data_matrix = masked_data_matrix[non_zeros_indexes,:]
        #print non_zeros_data_matrix.shape
                
        #mean_non_zeros_data_matrix = np.mean(non_zeros_data_matrix,axis = 0)
        
        #print mean_non_zeros_data_matrix.shape
    #else:
        #print "Warning, Image and mask are incompatible"
        #print img_shape
        #print data_mask
        #return
    
    #return mean_non_zeros_data_matrix
    
#def mean_select_non_null_data(data_img,data_mask):
      
    #img_shape = data_img.shape
    #mask_shape = data_mask.shape
    
    #print img_shape
    #print mask_shape
    
    #coords = return_coord_non_null_mask(data_mask)
    
    ##print coords
    ##print coords.shape
    
    ##print data_mask.size
    ##print np.sum(data_mask == 1)
    
    #if img_shape[:3] == mask_shape:
        #print "Image and mask are compatible"
        
        #masked_data_matrix = data_img[data_mask > min_non_null_mask_value,:]
        ##print masked_data_matrix.shape
       
        
        #non_zeros_indexes = np.sum(masked_data_matrix < min_BOLD_intensity,axis = 1) == 0
        #print non_zeros_indexes.shape
        
        #non_zeros_data_matrix = masked_data_matrix[non_zeros_indexes,:]
        #print non_zeros_data_matrix.shape
                
        #mean_non_zeros_data_matrix = np.mean(non_zeros_data_matrix,axis = 0)
        
        #print mean_non_zeros_data_matrix.shape
    #else:
        #print "Warning, Image and mask are incompatible"
        #print img_shape
        #print data_mask
        #return
    
    #return mean_non_zeros_data_matrix
    
############################################# covariate regression ##############################

def regress_parameters(data_matrix,covariates):

    import statsmodels.formula.api as smf
    
    print data_matrix.shape
    
    print covariates.shape
    
    resid_data = []
    
    data_names = ['Var_' + str(i) for i in range(data_matrix.shape[0])]
    
    print data_names
    
    covar_names = ['Cov_' + str(i) for i in range(covariates.shape[1])]
    
    print np.transpose(data_matrix).shape
    
    all_data = np.concatenate((np.transpose(data_matrix),covariates), axis = 1)
    
    print all_data.shape
    
    col_names = data_names + covar_names
    
    df = pd.DataFrame(all_data, columns = col_names)
    
    print df
    
    
    for var in data_names:
        
        formula = var + " ~ " + " + ".join(covar_names)
            
        print formula
        
        est = smf.ols(formula=formula, data=df).fit()
        
        ##print est.summary()
        
        #print est.resid.values
        
        resid = est.resid.values        
        resid_data.append(est.resid.values)
        

    resid_data_matrix = np.array(resid_data,dtype = float)
    
    print resid_data_matrix.shape
    
    return resid_data_matrix

def regress_filter_normalize_parameters(data_matrix,covariates):

    import statsmodels.formula.api as smf
    
    print data_matrix.shape
    
    print covariates.shape
    
    resid_data = []
    resid_filt_data = []
    z_score_data = []
    
    b,a = filt.iirfilter(N = 5, Wn = 0.04, btype = 'highpass')
        
    data_names = ['Var_' + str(i) for i in range(data_matrix.shape[0])]
    
    print data_names
    
    
    covar_names = ['Cov_' + str(i) for i in range(covariates.shape[1])]
    
    
    print np.transpose(data_matrix).shape
    
    all_data = np.concatenate((np.transpose(data_matrix),covariates), axis = 1)
    
    print all_data.shape
    
    col_names = data_names + covar_names
    
    df = pd.DataFrame(all_data, columns = col_names)
    
    print df
    
    
    for var in data_names:
        
        formula = var + " ~ " + " + ".join(covar_names)
            
        print formula
        
        est = smf.ols(formula=formula, data=df).fit()
        
        ##print est.summary()
        
        #print est.resid.values
        
        resid = est.resid.values        
        resid_data.append(est.resid.values)
        
        resid_filt = filt.filtfilt(b,a,x = resid)
        resid_filt_data.append(resid_filt)
        
        z_score  = stats.zscore(resid_filt)
        z_score_data.append(z_score)


    resid_data_matrix = np.array(resid_data,dtype = float)
    
    print resid_data_matrix.shape
    
    resid_filt_data_matrix = np.array(resid_filt_data, dtype = float)
    
    #print resid_filt_data_matrix
    
    z_score_data_matrix = np.array(z_score_data, dtype = float)
    
    #print z_score_data_matrix
        
    return resid_data_matrix,resid_filt_data_matrix,z_score_data_matrix

def regress_parameters_rpy(data_matrix,covariates):

    import rpy
    
    resid_data_matrix = np.zeros(shape = data_matrix.shape,dtype = 'float')
    
    for i in range(data_matrix.shape[0]):
        
        rpy.r.assign('r_serie', data_matrix[i,:])
        rpy.r.assign('r_rp', covariates)
        
        r_formula = "r_serie ~"
        
        print covariates.shape
        
        for cov in range(covariates.shape[1]):
        
            r_formula += " r_rp[,"+str(cov+1)+"]"
            
            if cov != range(covariates.shape[1])[-1]:
            
                r_formula += " +"
            
        resid_data_matrix[i,] = rpy.r.lm(rpy.r(r_formula))['residuals']
        
        print resid_data_matrix[i,]
        
    print resid_data_matrix
    
    return resid_data_matrix

def regress_filter_normalize_parameters_rpy(data_matrix,covariates):

    import rpy
    
    resid_data_matrix = np.zeros(shape = data_matrix.shape)
    resid_filt_data_matrix = np.zeros(shape = data_matrix.shape)
    z_score_data_matrix = np.zeros(shape = data_matrix.shape)
    
    b,a = filt.iirfilter(N = 5, Wn = 0.04, btype = 'highpass')
        
    #for i in [0,1]:
    for i in range(data_matrix.shape[0]):
        
        #print data_matrix[i,:].shape
        
        #print covariates.shape
        
        #print i
        rpy.r.assign('r_serie', data_matrix[i,:])
        rpy.r.assign('r_rp', covariates)
        
        r_formula = "r_serie ~"
        
        print covariates.shape
        
        for cov in range(covariates.shape[1]):
        
            r_formula += " r_rp[,"+str(cov+1)+"]"
            
            if cov != range(covariates.shape[1])[-1]:
            
                r_formula += " +"
            
        
        #print r_formula
        
        resid_data_matrix[i,] = rpy.r.lm(rpy.r(r_formula))['residuals']
        
        resid_filt_data_matrix[i,] = filt.lfilter(b,a,x = resid_data_matrix[i,])
        
        z_score_data_matrix[i,] = stats.zscore(resid_filt_data_matrix[i,] )

    return resid_data_matrix,resid_filt_data_matrix,z_score_data_matrix


#def regress_movement_parameters_rpy2(data_matrix,rp):
    
    #import rpy
    #import rpy2

    ##from rpy2.robjects import FloatVector
    ##from rpy2.robjects.packages import importr

    #resid_data_matrix = np.zeros(shape = data_matrix.shape)

    ##for i in [0,1]:
    #for i in range(data_matrix.shape[0]):
                
        #print i
        #print data_matrix[i,:]
        #print rp
        
        ##stats = importr('stats')
        ##base = importr('base')

        ##ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
        ##trt = FloatVector([4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69])
        ##group = base.gl(2, 10, 20, labels = ["Ctl","Trt"])
        ##weight = ctl + trt

        ##robjects.globalenv["weight"] = weight
        ##robjects.globalenv["group"] = group
        ##lm_D9 = stats.lm("weight ~ group")
        ##print(stats.anova(lm_D9))

        ### omitting the intercept
        ##lm_D90 = stats.lm("weight ~ group - 1")
        ##print(base.summary(lm_D90))

        #rpy.r.assign('r_serie', data_matrix[i,:])
        #rpy.r.assign('r_rp', rp)
        
        #print 'after assign'
        #resid = rpy.r.lm(rpy.r("r_serie ~  r_rp[,1] + r_rp[,2] + r_rp[,3] + r_rp[,4] + r_rp[,5] + r_rp[,6]"))['residuals']
        
        #print 'after lm'
         
        ##resid = regress_serie_movement_parameters(data_matrix[i,:],rp)
        #resid_data_matrix[i,] = stats.zscore(resid)

    #print resid_data_matrix.shape
    #return resid_data_matrix

######################################### euclidean distance between all coordinates
#def compute_dist_matrix(coords):

    #print coords.shape
    
    #dist_matrix = dist.squareform(dist.pdist(coords,metric='euclidean'))
    
    #print dist_matrix.shape
    
    #return dist_matrix
    
    
########################################## computation on correlation matrix
    
    ######## version sans std et var
#def return_Z_cor_mat(ts_mat,regressor_vect):
    
    #t1 = time.time()
    
    ##print regressor_vect
    
    ##print ts_mat
    
    #print ts_mat.shape[0],len(regressor_vect)
    
    #if ts_mat.shape[0] != len(regressor_vect):
        #print "Warning, incompatible regressor length {} {}".format(ts_mat.shape[0], len(regressor_vect))
        #return

    
    #keep = regressor_vect > 0.0
    
    #print np.sum(keep)
    
    #regressor_vect = regressor_vect[keep]
    #ts_mat = ts_mat[keep,:]
    
    #w = regressor_vect
    #s, n = ts_mat.shape
    
    #Z_cor_mat = np.zeros((n,n),dtype = float)
    
    #ts_mat2 = ts_mat*np.sqrt(w)[:,np.newaxis]
    
    #for i,j in it.combinations(range(n), 2):
    
        #s1 = ts_mat2[:,i]
        #s2 = ts_mat2[:,j]
        
        #Z_cor_mat[i,j] = np.arctanh((s1*s2).sum()/np.sqrt((s1*s1).sum() *(s2*s2).sum()))
        
    #t2 = time.time()
    
    #print "Weighted correlation computation took " + str(t2-t1) + "s"
    
    #return Z_cor_mat
    
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
    
    
#def return_var_cor_mat(ts_mat,regressor_vect):
    
    #t1 = time.time()
    
    #### ne marche pas, merci Sam...
    ##assert ts_mat.shape[0] == len(regressor_vect), "Warning, incompatible regressor length {} {}".format(ts_mat.shape[0], len(regressor_vect))

    ##print regressor_vect
    
    ##print ts_mat
    
    #if ts_mat.shape[0] != len(regressor_vect):
        #"Warning, incompatible regressor length {} {}".format(ts_mat.shape[0], len(regressor_vect))
        #return

    
    #keep = regressor_vect > 0.0
    
    ##print keep
    
    #regressor_vect = regressor_vect[keep]
    #ts_mat = ts_mat[keep,:]
    
    #w = regressor_vect
    #s, n = ts_mat.shape
    
    #Z_cor_mat = np.zeros((n,n),dtype = float)
    #var_cor_mat = np.zeros((n,n),dtype = float)
    #p_cor_mat = np.zeros((n,n),dtype = float)
    
    #rpy.r.library('weights')
        
    #t3 = time.time()
    
    ##print "Weighted correlation with weights R package computation took " + str(t3-t2) + "s"
    
    ##rpy.r.assign('s',ts_mat[:,:100])
    #rpy.r.assign('s',ts_mat)
    
    #rpy.r.assign('weights',w)
    
    #wei_cor = rpy.r("wtd.cor(as.matrix(s),as.matrix(s),weight = weights,collapse = FALSE)")
    
    #print wei_cor
        
        ###print wei_cor[0],Z_val

        
    #cor_mat = np.array(wei_cor['correlation'])
    #sderr_cor_mat =  np.array(wei_cor['std.err'])
    #pval_cor_mat = np.array(wei_cor['p.value'])
    
    #t4 = time.time()
    
    #print "Weighted correlation with weights R package computation took " + str(t4-t3) + "s"
    
    
    
    #return cor_mat,sderr_cor_mat,pval_cor_mat 
    
############### return cor list (raw)


#def return_net_list(Z_cor_mat):

    #t2 = time.time()
    
    #print Z_cor_mat.shape 
    
    #x_sig,y_sig = np.where(Z_cor_mat != 0.0)
        
    #net_list = np.array(np.column_stack((x_sig + 1,y_sig + 1,Z_cor_mat[x_sig,y_sig]*1000)),dtype = int)
    
    #print net_list.shape
    
    #t3 = time.time()
    
    #print "Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s"
    
    #return net_list
    
############### return cor list after thresholding and/or distance removal


#def return_signif_conf_net_list(cor_mat,conf_cor_mat):

    #t2 = time.time()
    
    #print cor_mat.shape 
    
    #binary_signif_matrix = np.absolute(cor_mat) >  np.absolute(conf_cor_mat)
    
    #print binary_signif_matrix.shape
    
    #x_sig,y_sig = np.where(binary_signif_matrix == 1)
        
    #net_list = np.array(np.column_stack((x_sig + 1,y_sig + 1,cor_mat[x_sig,y_sig]*1000)),dtype = int)
    
    #print net_list.shape
    
    #t3 = time.time()
    
    #print "Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s"
    
    #return net_list,binary_signif_matrix
    

#def return_dist_thr_conf_net_list(cor_mat,conf_cor_mat,dist_matrix,cor_conf_thr,min_dist_between_voxels = 0):

    #t2 = time.time()
    
    #x_sig,y_sig = np.where((dist_matrix > min_dist_between_voxels) & (np.absolute(cor_mat) -  np.absolute(conf_cor_mat) > cor_conf_thr))
        
    #net_list = np.array(np.column_stack((x_sig + 1,y_sig + 1,cor_mat[x_sig,y_sig]*1000)),dtype = int)
    
    #print net_list.shape
    
    #t3 = time.time()
    
    #print "Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s"
    
    #return net_list
    
############### return Z list after thresholding and/or distance removal


#def return_dist_thr_net_list(Z_cor_mat,dist_matrix,min_Z = 0.0, min_dist = 0.0):

    #t2 = time.time()
    
    #x_sig,y_sig = np.where((dist_matrix > min_dist) & (np.absolute(Z_cor_mat) > min_Z))
        
    #net_list = np.array(np.column_stack((x_sig + 1,y_sig + 1,Z_cor_mat[x_sig,y_sig]*1000)),dtype = int)
    
    #print net_list.shape
    
    #t3 = time.time()
    
    #print "Sparse Weighted correlation thresholding computation took " + str(t3-t2) + "s"
    
    #return net_list
        
######################## weighted correlation matrix
    
#def compute_weighted_cor_mat_non_zeros(ts_mat,regressor_vect):
    
    #if ts_mat.shape[0] != len(regressor_vect):
        #print "Warning, incompatible regressor length " + str(len(regressor_vect)) + " with time series length " + str(ts_mat.shape[0]) 
        #return 
        
    #keep = regressor_vect > 0.0
    #regressor_vect = regressor_vect[keep]
    #ts_mat = ts_mat[keep,:]
    
    #w = regressor_vect
    #s, n = ts_mat.shape
    
    #Z_list = []
    
    #ts_mat2 = ts_mat*np.sqrt(w)[:,np.newaxis]
    
    #cor_mat = np.zeros((ts_mat.shape[1],ts_mat.shape[1]))
   
    #Z_cor_mat = np.zeros((ts_mat.shape[1],ts_mat.shape[1]))
    
    #for i,j in it.combinations(range(n), 2):
        
        
        #s1 = ts_mat2[:,i]
        #s2 = ts_mat2[:,j]
    
        #cor_val = (s1*s2).sum()/np.sqrt((s1*s1).sum() *(s2*s2).sum())

        #Z_val = np.arctanh(cor_val)
        ##Z_cor_mat = 
        ##print cor_val

        #cor_mat[i,j] =  cor_val
        #cor_mat[j,i] =  cor_val

        #Z_cor_mat[i,j] =  Z_val
        #Z_cor_mat[j,i] =  Z_val
        
    #return cor_mat,Z_cor_mat
     
################# module representaion in brain space 

#def return_mod_mask(mod_vect,coords,mask_shape):
    #import numpy as np

    #print mask_shape

    #if coords.shape[0] != mod_vect.shape[0]:
        #print "Warning, incompatible size between coords {} and mod vect {}".format(coords.shape[0],mod_vect.shape[0])
    
    
    #print coords.shape

    ##print coord[1,:]
    ##mod_mask = np.empty(mask_shape)
    #mod_mask = np.zeros(mask_shape)

    ##print mod_mask


    ##print coords

    #print np.unique(mod_vect)

    #for mod_index in np.unique(mod_vect):
    ##for mod_index in [0]:

        #print mod_index
        #mod_coord = coords[mod_vect == mod_index,:]

        ##print mod_coord
        #mod_coord_x,mod_coord_y,mod_coord_z = mod_coord[:,0],mod_coord[:,1],mod_coord[:,2]

        #mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ] = mod_index +1
        
        ##print mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ]
        
        
    ##print np.column_stack(np.where(mod_mask == 4)).shape

    #return mod_mask

    
#def return_mod_mask_corres_rel_coords_neighbourhood(mod_vect,node_rel_coords,data_mask_shape):
    
    #import numpy as np

    #import itertools as iter
    
    #old_coords = np.transpose(np.array(np.vstack((node_rel_coords[:,0] * data_mask_shape[0],node_rel_coords[:,1] * data_mask_shape[1],node_rel_coords[:,2] * data_mask_shape[2])),dtype = 'int64'))
    
    #print old_coords
    #print old_coords.shape

    #print np.min(old_coords[:,0]),np.max(old_coords[:,0])
    #print np.min(old_coords[:,1]),np.max(old_coords[:,1])
    #print np.min(old_coords[:,2]),np.max(old_coords[:,2])
    
    #print data_mask_shape
    
    #print node_rel_coords[old_coords[:,2] == 128,:]
    
    #for i,max in enumerate(data_mask_shape):
        #old_coords[old_coords[:,i] == data_mask_shape[i],i] = data_mask_shape[i] - 1
        
    
    #print np.min(old_coords[:,0]),np.max(old_coords[:,0])
    #print np.min(old_coords[:,1]),np.max(old_coords[:,1])
    #print np.min(old_coords[:,2]),np.max(old_coords[:,2])
    
    ##print node_corres
    ##print node_corres.shape
    
    ##old_coords = coords[node_corres,:]
    
    ##print old_coords.shape
    
    #if old_coords.shape[0] != mod_vect.shape[0]:
        #print "Warning, incompatible size between old_coords {} and mod vect {}".format(old_coords.shape[0],mod_vect.shape[0])
    
    ##print coord[1,:]
    ##mod_mask = np.empty(mask_shape)
    
    #neighbourhood = 3
    
    #mod_mask = np.zeros((data_mask_shape[0]+1 +2*neighbourhood,data_mask_shape[1]+1 +2*neighbourhood,data_mask_shape[2]+1 +2*neighbourhood),dtype = 'int')

    ##print mod_mask


    ##print coords

    ##print np.unique(mod_vect)

    
    #neigh_range = range(-neighbourhood,neighbourhood+1)
    
    #for mod_index in np.unique(mod_vect):
    ##for mod_index in [0]:

        ##print mod_index
        #mod_coord = old_coords[mod_vect == mod_index,:]

        ##print mod_coord
        #mod_coord_x,mod_coord_y,mod_coord_z = mod_coord[:,0],mod_coord[:,1],mod_coord[:,2]

        
        
        #for relative_coord in iter.product(neigh_range, repeat=3):

            #neigh_x = mod_coord_x + relative_coord[0] + neighbourhood
            #neigh_y = mod_coord_y + relative_coord[1] + neighbourhood
            #neigh_z = mod_coord_z + relative_coord[2] + neighbourhood

            #mod_mask[neigh_x,neigh_y,neigh_z ] = mod_index +1
        
        ##print mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ]
        
    ##print np.column_stack(np.where(mod_mask == 4)).shape

    #return mod_mask

#def return_mod_mask_corres_rel_coords(mod_vect,node_rel_coords,data_mask_shape):
    
    #import numpy as np

    #old_coords = np.transpose(np.array(np.vstack((node_rel_coords[:,0] * data_mask_shape[0],node_rel_coords[:,1] * data_mask_shape[1],node_rel_coords[:,2] * data_mask_shape[2])),dtype = 'int64'))
    
    #print old_coords
    #print old_coords.shape

    #print np.min(old_coords[:,0]),np.max(old_coords[:,0])
    #print np.min(old_coords[:,1]),np.max(old_coords[:,1])
    #print np.min(old_coords[:,2]),np.max(old_coords[:,2])
    
    #print data_mask_shape
    
    #print node_rel_coords[old_coords[:,2] == 128,:]
    
    #for i,max in enumerate(data_mask_shape):
        #old_coords[old_coords[:,i] == data_mask_shape[i],i] = data_mask_shape[i] - 1
        
    
    #print np.min(old_coords[:,0]),np.max(old_coords[:,0])
    #print np.min(old_coords[:,1]),np.max(old_coords[:,1])
    #print np.min(old_coords[:,2]),np.max(old_coords[:,2])
    
    ##print node_corres
    ##print node_corres.shape
    
    ##old_coords = coords[node_corres,:]
    
    ##print old_coords.shape
    
    #if old_coords.shape[0] != mod_vect.shape[0]:
        #print "Warning, incompatible size between old_coords {} and mod vect {}".format(old_coords.shape[0],mod_vect.shape[0])
    
    ##print coord[1,:]
    ##mod_mask = np.empty(mask_shape)
    #mod_mask = np.zeros(data_mask_shape)

    ##print mod_mask


    ##print coords

    ##print np.unique(mod_vect)

    #for mod_index in np.unique(mod_vect):
    ##for mod_index in [0]:

        ##print mod_index
        #mod_coord = old_coords[mod_vect == mod_index,:]

        ##print mod_coord
        #mod_coord_x,mod_coord_y,mod_coord_z = mod_coord[:,0],mod_coord[:,1],mod_coord[:,2]

        #mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ] = mod_index +1
        
        ##print mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ]
        
    ##print np.column_stack(np.where(mod_mask == 4)).shape

    #return mod_mask

    
    
    
#def return_mod_mask_corres(mod_vect,node_corres,coords,mask_shape):
    
    #import numpy as np

    #print mask_shape

    #print coords.shape

    #print node_corres
    #print node_corres.shape
    
    #old_coords = coords[node_corres,:]
    
    #print old_coords.shape
    
    #if old_coords.shape[0] != mod_vect.shape[0]:
        #print "Warning, incompatible size between old_coords {} and mod vect {}".format(old_coords.shape[0],mod_vect.shape[0])
    
    ##print coord[1,:]
    ##mod_mask = np.empty(mask_shape)
    #mod_mask = np.zeros(mask_shape)

    ##print mod_mask


    ##print coords

    ##print np.unique(mod_vect)

    #for mod_index in np.unique(mod_vect):
    ##for mod_index in [0]:

        ##print mod_index
        #mod_coord = old_coords[mod_vect == mod_index,:]

        ##print mod_coord
        #mod_coord_x,mod_coord_y,mod_coord_z = mod_coord[:,0],mod_coord[:,1],mod_coord[:,2]

        #mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ] = mod_index +1
        
        ##print mod_mask[mod_coord_x,mod_coord_y,mod_coord_z ]
        
    ##print np.column_stack(np.where(mod_mask == 4)).shape

    #return mod_mask

#def compute_average_ts_by_module_corres(ts_mat,coords,mod_vect,node_corres):

    #import numpy as np

    #print ts_mat.shape

    #print coords.shape

    #old_coords = coords[node_corres,:]
    
    #old_ts_mat = ts_mat[node_corres,:]
    
    ##print coord[1,:]
    ##print coords

    ##print np.unique(mod_vect)

    ##print np.max(mod_vect)

    #nb_mod = len(range(np.max(mod_vect) + 1))
    
    #print "nb_mod: " + str(nb_mod)

    #if len(np.unique(mod_vect)) != nb_mod :
        #print "warning, some module indexes are empty"

    #mod_average_ts_mat = np.empty((nb_mod,old_ts_mat.shape[1]))
    #mod_average_coords = np.empty((nb_mod,3))

    #for mod_index in np.unique(mod_vect):
    ##for mod_index in [0]:

        ##print mod_index

        #mod_indexes = mod_vect == mod_index

        #mod_coords = old_coords[mod_indexes,:]

        ##print mod_coords.shape

        #mod_ts_mat = old_ts_mat[mod_indexes,:]

        ##print mod_ts_mat.shape

        #mod_average_coords[mod_index,:] = np.mean(mod_coords,axis = 0)
        #mod_average_ts_mat[mod_index,:] = np.mean(mod_ts_mat,axis = 0)
        
    ##print np.column_stack(np.where(mod_mask == 4)).shape


    #return mod_average_ts_mat,mod_average_coords
    

##def compute_average_ts_by_module(ts_mat,coords,mod_vect):

    ##import numpy as np

    ##print ts_mat.shape

    ##print coords.shape

    ###print coord[1,:]
    ###print coords

    ##print np.unique(mod_vect)

    ###print np.max(mod_vect)

    ##nb_mod = len(range(np.max(mod_vect) + 1))
    
    ##print "nb_mod: " + str(nb_mod)

    ##if len(np.unique(mod_vect)) != nb_mod :
        ##print "warning, some module indexes are empty"

    ##mod_average_ts_mat = np.empty((nb_mod,ts_mat.shape[1]))
    ##mod_average_coords = np.empty((nb_mod,3))

    ##for mod_index in np.unique(mod_vect):
    ###for mod_index in [0]:

        ###print mod_index

        ##mod_indexes = mod_vect == mod_index

        ##mod_coords = coords[mod_indexes,:]

        ###print mod_coords.shape

        ##mod_ts_mat = ts_mat[mod_indexes,:]

        ###print mod_ts_mat.shape

        ##mod_average_coords[mod_index,:] = np.mean(mod_coords,axis = 0)
        ##mod_average_ts_mat[mod_index,:]  = np.mean(mod_ts_mat,axis = 0)
        
    ###print np.column_stack(np.where(mod_mask == 4)).shape


    ##return mod_average_ts_mat,mod_average_coords

    
####################################################### confusion matrices ################################################
    
#def return_confusion_matrix(community_vect1, coords_corres1, community_vect2, coords_corres2):
    
    #print 'Total number of nodes in partitions: ' + str(coords_corres1.shape[0]) + ' ' + str(coords_corres2.shape[0])
    
    ############### computing number of node in common
    
    #nb_common_nodes = compute_number_common_nodes(coords_corres1.tolist(),coords_corres2.tolist())
    
    #print 'Total number of nodes in common = ' + str(nb_common_nodes) 
    
    ############## Checking vector homogeniety size, and modular indexation
    #if len(community_vect1) != coords_corres1.shape[0] :
        #print "warning, number of nodes in node_corres1 and community_vect1 are imcompatible {} {}".format(len(community_vect1),coords_corres1.shape[0] )

    #if len(community_vect2) != coords_corres2.shape[0] :
        #print "warning, number of nodes in node_corres2 and community_vect2 are imcompatible {} {}".format(len(community_vect2),coords_corres2.shape[0] )

    #nb_mod1 = np.max(community_vect1) + 1
    
    #nb_mod2 = np.max(community_vect2) + 1
    
    #if len(np.unique(community_vect1)) != nb_mod1 :
        #print "warning, number of modules community_vect1 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect1)),nb_mod1)

    #if len(np.unique(community_vect2)) != nb_mod2 :
        #print "warning, number of modules community_vect2 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect2)),nb_mod2)

    #print "nb_mod1: " + str(nb_mod1)
    
    #print "nb_mod2: " + str(nb_mod2)
    
    ########## computing confusion matrix
    
    #confusion_matrix = np.zeros((nb_mod1,nb_mod2),dtype = 'uint64')
        
    #for part1_mod_index in np.unique(community_vect1):   
    ##for part1_mod_index in range(5):
        
        #part1_mod_coords = coords_corres1[community_vect1 == part1_mod_index,:].tolist()

        #for part2_mod_index in np.unique(community_vect2):
    
            #part2_mod_coords = coords_corres2[community_vect2 == part2_mod_index,:].tolist()

            #confusion_matrix[part1_mod_index,part2_mod_index] = compute_number_common_nodes(part1_mod_coords,part2_mod_coords)
            
    #print confusion_matrix
    
    ################ resorting nodes according to best partitions matches
    
    ##max_match_1 = np.argmax(confusion_matrix,axis = 1)
    
    ##print max_match_1
    
    ##print confusion_matrix[range(nb_mod1),max_match_1]
    
    ##max_match_2 = np.argmax(confusion_matrix,axis = 0)
    
    ##print max_match_2
    
    ##print confusion_matrix[max_match_2,range(nb_mod2)]
    
    ##print np.sum(confusion_matrix[range(nb_mod1),max_match_1]),np.sum(confusion_matrix[max_match_2,range(nb_mod2)])
    
    ############### printing modules with highest matches
    
    #sim_mod1,sim_mod2 = np.where(confusion_matrix != 0)
    
    #ind_sorted = np.argsort(confusion_matrix[sim_mod1,sim_mod2])[::-1]
    
    #count_nodes1 = np.bincount(community_vect1)
    #count_nodes2 = np.bincount(community_vect2)
    
    #np.savetxt(sys.stdout,np.column_stack((sim_mod1,sim_mod2,count_nodes1[sim_mod1],count_nodes2[sim_mod2],confusion_matrix[sim_mod1,sim_mod2]))[ind_sorted[:20],:], '%d')
       
    #return confusion_matrix
        
        
#def return_confusion_matrix_dartel(community_vect1, coords_corres1, community_vect2, coords_corres2, gm_mask_coords):
    
    #print 'Total number of nodes in partitions: ' + str(coords_corres1.shape[0]) + ' ' + str(coords_corres2.shape[0])
    
    ############### computing number of node in common
    
    #nb_common_nodes = compute_number_common_nodes(coords_corres1.tolist(),coords_corres2.tolist())
    
    #print 'Total number of nodes in common between partitions = ' + str(nb_common_nodes) 
    
    #nb_common_nodes_gm1 = compute_number_common_nodes(coords_corres1.tolist(),gm_mask_coords.tolist())
    
    #nb_common_nodes_gm2 = compute_number_common_nodes(coords_corres2.tolist(),gm_mask_coords.tolist())
    
    #print 'Total number of nodes in common with gm mask = ' + str(nb_common_nodes_gm1) + ' ' + str(nb_common_nodes_gm2) 
    
    
    ############### Checking vector homogeniety size, and modular indexation
    #if len(community_vect1) != coords_corres1.shape[0] :
        #print "warning, number of nodes in node_corres1 and community_vect1 are imcompatible {} {}".format(len(community_vect1),coords_corres1.shape[0] )

    #if len(community_vect2) != coords_corres2.shape[0] :
        #print "warning, number of nodes in node_corres2 and community_vect2 are imcompatible {} {}".format(len(community_vect2),coords_corres2.shape[0] )

    #nb_mod1 = np.max(community_vect1) + 1
    
    #nb_mod2 = np.max(community_vect2) + 1
    
    #if len(np.unique(community_vect1)) != nb_mod1 :
        #print "warning, number of modules community_vect1 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect1)),nb_mod1)

    #if len(np.unique(community_vect2)) != nb_mod2 :
        #print "warning, number of modules community_vect2 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect2)),nb_mod2)

    ##print "nb_mod1: " + str(nb_mod1)
    
    ##print "nb_mod2: " + str(nb_mod2)
    
    ########## computing confusion matrix
    
    #confusion_matrix = np.zeros((nb_mod1,nb_mod2),dtype = 'uint64')
        
    #for part1_mod_index in np.unique(community_vect1):   
    ##for part1_mod_index in range(5):
        
        #part1_mod_coords = coords_corres1[community_vect1 == part1_mod_index,:].tolist()

        #for part2_mod_index in np.unique(community_vect2):
    
            #part2_mod_coords = coords_corres2[community_vect2 == part2_mod_index,:].tolist()

            #confusion_matrix[part1_mod_index,part2_mod_index] = compute_number_common_nodes(part1_mod_coords,part2_mod_coords)
            
    ##print confusion_matrix
    
    ################ resorting nodes according to best partitions matches
    
    ##max_match_1 = np.argmax(confusion_matrix,axis = 1)
    
    ##print max_match_1
    
    ##print confusion_matrix[range(nb_mod1),max_match_1]
    
    ##max_match_2 = np.argmax(confusion_matrix,axis = 0)
    
    ##print max_match_2
    
    ##print confusion_matrix[max_match_2,range(nb_mod2)]
    
    ##print np.sum(confusion_matrix[range(nb_mod1),max_match_1]),np.sum(confusion_matrix[max_match_2,range(nb_mod2)])
    
    ############### printing modules with highest matches
    
    #sim_mod1,sim_mod2 = np.where(confusion_matrix != 0)
    
    #ind_sorted = np.argsort(confusion_matrix[sim_mod1,sim_mod2])[::-1]
    
    #count_nodes1 = np.bincount(community_vect1)
    #count_nodes2 = np.bincount(community_vect2)
    
    #np.savetxt(sys.stdout,np.column_stack((sim_mod1,sim_mod2,count_nodes1[sim_mod1],count_nodes2[sim_mod2],confusion_matrix[sim_mod1,sim_mod2]))[ind_sorted[:20],:], '%d')
       
    #return confusion_matrix
    
    
#def compute_number_common_nodes(coords_corres1,coords_corres2):

    #nb_common_coords = 0
    
    #for coord in coords_corres1:
        ##print coord
        #if coord in coords_corres2:
            
            ##print coord,coords_corres1.index(coord),coords_corres2.index(coord)
            #nb_common_coords+=1
            
    #return nb_common_coords
    
#def return_img(mask_file,coords,data_vect):

    #data_vect = np.array(data_vect)
    
    #mask = nib.load(mask_file)
    
    #mask_data_shape = mask.get_data().shape
    
    ##print mask_data_shape
    
    #mask_header = mask.get_header()
    
    #mask_affine = mask.get_affine()
    
    ##print mask_affine
    
    #if np.any(mask.shape < np.amax(coords,axis = 0)):
        #print "warning, mask shape not compatible with coords  {} {}".format(mask.shape,np.amax(coords,axis = 0))
        
    #if coords.shape[0] < data_vect.shape[0]:
        #print "warning, coords not compatible with data_vect: {} {}".format(coords.shape[0], data_vect.shape[0])
        
    #data = np.zeros((mask.shape),dtype = data_vect.dtype)
    
    #data[coords[:,0],coords[:,1],coords[:,2]] = data_vect

    ##print 'data img'
    #data_img = nib.Nifti1Image(data,mask_affine,mask_header)
    
    #return data_img
    
    
################################### Gather partitions ###################################################################################################
    
#def return_gather_partitions(community_vect1,coords_corres1,community_vect2,coords_corres2,gm_mask_data_shape,confusion_matrix):

    #print 'Total number of nodes in partitions: ' + str(coords_corres1.shape[0]) + ' ' + str(coords_corres2.shape[0])
    
    ############### computing number of node in common
    
    #nb_common_nodes = compute_number_common_nodes(coords_corres1.tolist(),coords_corres2.tolist())
    
    #print 'Total number of nodes in common between partitions = ' + str(nb_common_nodes) 
        
    ############### Checking vector homogeniety size, and modular indexation
    #if len(community_vect1) != coords_corres1.shape[0] :
        #print "warning, number of nodes in node_corres1 and community_vect1 are imcompatible {} {}".format(len(community_vect1),coords_corres1.shape[0] )

    #if len(community_vect2) != coords_corres2.shape[0] :
        #print "warning, number of nodes in node_corres2 and community_vect2 are imcompatible {} {}".format(len(community_vect2),coords_corres2.shape[0] )

    #nb_mod1 = np.max(community_vect1) + 1
    
    #nb_mod2 = np.max(community_vect2) + 1
    
    #if len(np.unique(community_vect1)) != nb_mod1 :
        #print "warning, number of modules community_vect1 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect1)),nb_mod1)

    #if len(np.unique(community_vect2)) != nb_mod2 :
        #print "warning, number of modules community_vect2 is irregular (some modules are missing) {} {}".format(len(np.unique(community_vect2)),nb_mod2)

    ##print "nb_mod1: " + str(nb_mod1)
    
    ##print "nb_mod2: " + str(nb_mod2)
    
    ########## computing confusion matrix
    
    #print confusion_matrix
    
    #most_similar_part1 = np.transpose(np.vstack((range(nb_mod1),np.argmax(confusion_matrix,axis= 1)))).tolist()
    
    #print most_similar_part1
    
    #most_similar_part2 = np.transpose(np.vstack((np.argmax(confusion_matrix,axis= 0),range(nb_mod2)))).tolist()
    
    #print most_similar_part2
    
    
    #gather_partitions_data = np.zeros(gm_mask_data_shape,dtype = 'int64')
       
    
    #most_similar_part = [list(x) for x in set(tuple(x) for x in most_similar_part1 + most_similar_part2)]
    
    #print most_similar_part
    
    ###.index_array
    ###most_similar_part2, = np.where(np.max(confusion_matrix,axis= 1))
    
    ##print most_similar_part1
    
    ##print most_similar_part2
    
    #sum_inter = 0
    
    #for i in range(len(most_similar_part)):
    
        
        #pair_part = most_similar_part[i]
        #part1_mod_coords = coords_corres1[community_vect1 == pair_part[0],:]
        #list_part1_mod_coords = part1_mod_coords.tolist()
        
        #part2_mod_coords = coords_corres2[community_vect2 == pair_part[1],:]
        #list_part2_mod_coords = part2_mod_coords.tolist()

        #inter_set = intersect_common_nodes(list_part1_mod_coords,list_part2_mod_coords)
        
        #sum_inter += inter_set.shape[0]
        
        #print pair_part[0],pair_part[1],part1_mod_coords.shape[0],part2_mod_coords.shape[0],inter_set.shape[0]
        
        #gather_partitions_data[inter_set[:,0],inter_set[:,1],inter_set[:,2]] = i
        
        
    ##sum_inter1 = 0
    
    ##for i in range(len(most_similar_part2)):
    
        ##part1_mod_coords = coords_corres1[community_vect1 == i,:]
        ##list_part1_mod_coords = part1_mod_coords.tolist()
        
        ##part2_mod_coords = coords_corres2[community_vect2 == most_similar_part2[i],:]
        ##list_part2_mod_coords = part2_mod_coords.tolist()

        ##inter_set = intersect_common_nodes(list_part1_mod_coords,list_part2_mod_coords)
        
        ##sum_inter1 += inter_set.shape[0]
        
        ##print i,most_similar_part2[i],part1_mod_coords.shape[0],part2_mod_coords.shape[0],inter_set.shape[0]
        
        ##gather_partitions_data[inter_set[:,0],inter_set[:,1],inter_set[:,2]] = -i
        
        
    ##for part1_mod_index in np.unique(community_vect1):   
    ####for part1_mod_index in range(5):
        
        ##part1_mod_coords = coords_corres1[community_vect1 == part1_mod_index,:]
        ##list_part1_mod_coords = part1_mod_coords.tolist()

        ##for part2_mod_index in np.unique(community_vect2):
    
            ##part2_mod_coords = coords_corres2[community_vect2 == part2_mod_index,:]
            ##list_part2_mod_coords = part2_mod_coords.tolist()

            ##### compute commun set
            
            
            ##inter_set,exclu_set = intersect_exclusive_common_nodes(list_part1_mod_coords,list_part2_mod_coords)
            
            ##sum_inter += inter_set.shape[0]
            
            ##print part1_mod_coords.shape[0],part2_mod_coords.shape[0],part1_mod_index,part2_mod_index,inter_set.shape[0],exclu_set.shape[0]
            
            ##gather_partitions_data[inter_set[:,0],inter_set[:,1],inter_set[:,2]] += 1
            
            ##gather_partitions_data[exclu_set[:,0],exclu_set[:,1],exclu_set[:,2]] += -1
            
     
    #print sum_inter
    ##print gather_partitions_data
    
    #return gather_partitions_data
    

#def intersect_common_nodes(coords_corres1,coords_corres2):

    #intersect_list = []
    
    #for coord in coords_corres1:
        ##print coord
        #if coord in coords_corres2:
            
            ##print coord,coords_corres1.index(coord),coords_corres2.index(coord)
            #intersect_list.append(coord)
        
            
    #return np.array(intersect_list,dtype = int)


    
    
    
#def intersect_exclusive_common_nodes(coords_corres1,coords_corres2):

    #intersect_list = []
    #exclusive_list = []
    
    #for coord in coords_corres1:
        ##print coord
        #if coord in coords_corres2:
            
            ##print coord,coords_corres1.index(coord),coords_corres2.index(coord)
            #intersect_list.append(coord)
        #else:
            #exclusive_list.append(coord)
            
    #for coord in coords_corres2:
        #if not (coord in coords_corres1 and coord in exclusive_list):
            #exclusive_list.append(coord)
    #return np.array(intersect_list,dtype = int),np.array(exclusive_list,dtype = int)
    
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
    
    

#def return_coclass_mat_list_by_module(community_vect,corres_coords,gm_mask_coords):

    #print corres_coords.shape[0],community_vect.shape[0]
    
    #if (corres_coords.shape[0] != community_vect.shape[0]):
        #print "warning, length of corres_coords and community_vect are imcompatible {} {}".format(corres_coords.shape[0],community_vect.shape[0])
    
    ##print corres_coords
    
    ##print gm_mask_coords
    
    #where_in_gm = where_in_coords(corres_coords,gm_mask_coords)
    
    #print np.min(where_in_gm),np.max(where_in_gm),where_in_gm.shape
    
    
    
    
    ##find_in_corres = find_index_in_coords(gm_mask_coords,corres_coords)
    
    ##print find_in_corres
    
    #if (where_in_gm.shape[0] != community_vect.shape[0]):
        #print "warning, length of where_in_gm and community_vect are imcompatible {} {}".format(where_in_gm.shape[0],community_vect.shape[0])
    
    
    #list_module_indexes = np.unique(community_vect)
    
    #print list_module_indexes
    
    #coclass_mat_list = []
    
    #possible_edge_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        
    #for mod_index in list_module_indexes:
    
        #coclass_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
            
        ##print np.equal(community_vect,community_vect).shape
        
        ##print coclass_mat[where_in_gm][:,where_in_gm].shape
        
        #for i,j in it.combinations(range(where_in_gm.shape[0]),2):
        
            #if community_vect[i] == community_vect[j] and community_vect[i] == mod_index:
                
                #coclass_mat[where_in_gm[i],where_in_gm[j]] = coclass_mat[where_in_gm[j],where_in_gm[i]] = 1
                
            #possible_edge_mat[where_in_gm[i],where_in_gm[j]] = 1
            #possible_edge_mat[where_in_gm[j],where_in_gm[i]] = 1
            
        #coclass_mat_list.append(coclass_mat)
        
    #return coclass_mat_list,possible_edge_mat
    
    
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
    
    
    ################################################# gather correl mat #######################################
    
    #### from Z_list
#def return_dense_correl_mat(Z_list,coords,gm_mask_coords):
    
    #where_in_gm = where_in_coords(coords,gm_mask_coords)
    
    #print where_in_gm
        
    #correl_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        
    #for edge in Z_list.tolist():
    
        #i = edge[0] -1 
        #j = edge[1] -1
        #Z_cor = edge[2]
        
        ##print i,j,Z_cor
    
        #correl_mat[where_in_gm[i],where_in_gm[j]] = Z_cor
        ##correl_mat[where_in_gm[j],where_in_gm[i]] = Z_cor
        
    #return correl_mat
    
    
    #### from Z_cor_mat
    
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

##def return_corres_correl_mat(Z_cor_mat,coords,gm_mask_coords):
    
    ##where_in_gm = where_in_coords(coords,gm_mask_coords)
    
    
    ##print np.min(where_in_gm),np.max(where_in_gm),where_in_gm.shape
    
    ##print where_in_gm
        
    ##corres_correl_mat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = float)
        
    ##for i,j in it.combinations(range(len(where_in_gm)),2):
    
        ##corres_correl_mat[where_in_gm[i],where_in_gm[j]] = Z_cor_mat[i,j]
        ##corres_correl_mat[where_in_gm[j],where_in_gm[i]] = Z_cor_mat[i,j]
        
    ##return corres_correl_mat
    
###################################### Formatting data for external community detection algorithm (Louvain_Traag) ##############################
        
#def export_Louvain_net_from_list(Z_Louvain_file,Z_list,coords):
    
    #print np.array(Z_list).shape
    
    ##print sig_x,sig_y
    #print "column_stack"
    #tab_edges = np.column_stack((np.array(Z_list),np.repeat(1,repeats = len(Z_list))))
    
    #print tab_edges.shape
    
    #print "file"
    
    #with open(Z_Louvain_file,'w') as f:
        
        ##### write node list
        #nb_nodes = coords.shape[0]
        
        #print "Nb nodes: " + str(nb_nodes)
        
        #coords_list = coords.tolist()
        
        #f.write('>\n')
        
        #for node in range(nb_nodes):
            
            ###print node + coord label
            #f.write(str(node+1) + ' ' + '_'.join(map(str,coords_list[node])) + '\n')
        
        ##### write slice list
        #f.write('>\n')
        #f.write('1 1\n')
        
        ##### write edge list
        #f.write('>\n')
        
        #np.savetxt(f,tab_edges,fmt = '%d %d %d %d')
        
#def read_Louvain_net_file(Z_Louvain_file):
    
    ##print "file"
    
    #with open(Z_Louvain_file,'r') as f:
            
        
        #lines = [line.strip().split(' ') for line in f ]
    
        ##print lines
        
        #count_sign = []
        
        #for i,line in enumerate(lines):
            
            #if line == ['>']:
                #print i
                
                #count_sign.append(i)
            
            
        ##print count_sign
        
        ##print lines[count_sign[2]+1:]
        
        #edge_indexes = np.array(lines[count_sign[2]+1:],dtype = 'int64')[:,:3]
        
        ##print edge_indexes
        
        ##print edge_indexes.shape[0]
        
    #return edge_indexes
    
###################################### Formatting data for external community detection algorithm (radatools) ##############################
        
############# modular partition file
    
    
#### node correspondance files

#def read_Louvain_corres_nodes(Louvain_node_file):

    #data = np.loadtxt(Louvain_node_file,dtype={'names': ('old_index', 'label', 'slice','weight','Louvain_index'),'formats': ('i8', 'S8', 'i8', 'i8', 'i8')})

    #return data['old_index']-1
    
    
##################################################################### reordering #########################################################

#def force_order(community_vect,node_orig_indexes):
    
    #reordered_community_vect = np.zeros(shape = community_vect.shape, dtype = 'int')-1
    
    #list_used_mod_index = []
    
    ##print np.unique(community_vect)
    
    #nb_mod_init = np.unique(community_vect).shape[0]
    
    #index = 0
    
    #for i in np.unique(node_orig_indexes):
    
        #print "For spm contrast index " + str(i)
        
        #c = Counter(community_vect[node_orig_indexes == i])
        
        #nb_Rois_in_contrast = np.sum(node_orig_indexes == i)
        
        #for count in c.most_common():
        
            #(most_common_mod_index,most_common_number) = count
            
            #print most_common_mod_index
            
            #if most_common_mod_index != -1:
                    
                ##(most_common_mod_index,most_common_number) = c.most_common(1)[0]
                
                
                #print "Found module index %d represents %f of Rois in constrast" %(most_common_mod_index,float(most_common_number)/nb_Rois_in_contrast) 
                
                #reordered_community_vect[community_vect == most_common_mod_index] = index
                
                #list_used_mod_index.append(most_common_mod_index)
                
                #community_vect[community_vect == most_common_mod_index] = -1
                
                #index = index +1
                
                #break
                
    #print "after dist reorg"
    
    #print community_vect
    
    #for i in np.unique(community_vect):
    
        #if i == -1:
            
            #continue
        
        #if i in np.unique(reordered_community_vect):
        
            #print "Warning, %d is already present in reordered_community_vect"%(i)
            
        #else:
            
            #reordered_community_vect[community_vect == i] = index
            
            #index = index +1
            
    #print np.unique(reordered_community_vect)
    
    #nb_mod_fin = np.unique(reordered_community_vect).shape[0]
    
    #if nb_mod_init != nb_mod_fin:
        #print "Warning, nb_mod_init (%d) != nb_mod_fin (%d) "%(nb_mod_init,nb_mod_fin)
        
    #return reordered_community_vect
    