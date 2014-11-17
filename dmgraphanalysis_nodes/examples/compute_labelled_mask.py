# -*- coding: utf-8 -*-

"""
Compute ROI labeled mask from spm contrast image or images
"""


import sys, os
sys.path.append('../irm_analysis')

from  define_variables import *
from dmgraphanalysis_nodes.utils_dtype_coord import *
    
import glob

from xml.dom import minidom
import os 

import numpy as np

from nibabel import load, save

import nipy.labs.spatial_models.mroi as mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image
import nipy.labs.spatial_models.hroi as hroi

import nipy.labs.statistical_mapping as stat_map

import itertools as iter
    
import scipy.spatial.distance as dist


### scan toutes les possibilités dans le cube, et ne retourne que les ROIs dont le nombre de voxels dans le voisinage appartienant à AAL et au mask est supérieur à min_nb_voxels_in_neigh 
def return_indexed_mask_neigh_within_binary_template(peak_position,neighbourhood,resliced_template_data,orig_peak_coords_dt):
    
    peak_x,peak_y,peak_z = np.array(peak_position,dtype = 'int')
    
    neigh_range = range(-neighbourhood,neighbourhood+1)
    
    list_neigh_coords = []
    
    peak_template_roi_index = resliced_template_data[peak_x,peak_y,peak_z]
    
    #print "template index = " + str(peak_template_roi_index)
    
    count_neigh_in_orig_mask = 0
        
    if peak_template_roi_index != 0:
        
        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x,neigh_y,neigh_z = peak_position + relative_coord

            neigh_coord_dt = convert_np_coords_to_coords_dt(np.array([[neigh_x,neigh_y,neigh_z]]))
            #neigh_coord_dt = np.array([(neigh_x,neigh_y,neigh_z), ], dtype = coord_dt)
            
            neigh_template_roi_index = resliced_template_data[neigh_x,neigh_y,neigh_z]
            
            #print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
            #if neigh_template_roi_index == peak_template_roi_index and np.in1d(neigh_coord_dt,orig_peak_coords_dt):
            if neigh_template_roi_index != 0 and neigh_coord_dt in orig_peak_coords_dt:
            
                list_neigh_coords.append(np.array([neigh_x,neigh_y,neigh_z],dtype = 'int16'))
                
                count_neigh_in_orig_mask = count_neigh_in_orig_mask +1
                
        if min_nb_voxels_in_neigh <= len(list_neigh_coords):
            
            return list_neigh_coords,peak_template_roi_index
            
    return [],0
    
def return_indexed_mask_neigh_within_template(peak_position,neighbourhood,resliced_template_data,orig_peak_coords_dt):
    
    peak_x,peak_y,peak_z = np.array(peak_position,dtype = 'int')
    
    neigh_range = range(-neighbourhood,neighbourhood+1)
    
    list_neigh_coords = []
    
    peak_template_roi_index = resliced_template_data[peak_x,peak_y,peak_z]
    
    print "template index = " + str(peak_template_roi_index)
    
    count_neigh_in_orig_mask = 0
        
    if peak_template_roi_index != 0:
        
        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x,neigh_y,neigh_z = peak_position + relative_coord

            neigh_coord_dt = convert_np_coords_to_coords_dt(np.array([[neigh_x,neigh_y,neigh_z]]))
            #neigh_coord_dt = np.array([(neigh_x,neigh_y,neigh_z), ], dtype = coord_dt)
            
            neigh_template_roi_index = resliced_template_data[neigh_x,neigh_y,neigh_z]
            
            #print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
            #if neigh_template_roi_index == peak_template_roi_index and np.in1d(neigh_coord_dt,orig_peak_coords_dt):
            if neigh_template_roi_index == peak_template_roi_index and neigh_coord_dt in orig_peak_coords_dt:
            
                list_neigh_coords.append(np.array([neigh_x,neigh_y,neigh_z],dtype = 'int16'))
                
                count_neigh_in_orig_mask = count_neigh_in_orig_mask +1
                
        if min_nb_voxels_in_neigh <= len(list_neigh_coords):
            
            return list_neigh_coords,peak_template_roi_index
            
    return [],0
         
#def return_indexed_mask(peak_position,neighbourhood = 1):

    #neigh_range = range(-neighbourhood,neighbourhood+1)
    
    #list_neigh_coords = []
    
    #for relative_coord in iter.product(neigh_range, repeat=3):

        #neigh_coord = peak_position + relative_coord
        
        #list_neigh_coords.append(neigh_coord)
        
    #neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
    
    #return neigh_coords
    
def remove_close_peaks(list_orig_peak_coords,min_dist = 2.0 * np.sqrt(3)):
    
    list_selected_peaks_coords = []
    
    for orig_peak_coord in list_orig_peak_coords:
        
        orig_peak_coord_np = np.array(orig_peak_coord)
            
        if len(list_selected_peaks_coords) > 0:
            
            selected_peaks_coords_np = np.array(list_selected_peaks_coords)
            
            #orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)
            
            #selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)
            
            
            #print selected_peaks_coords_np.shape
            
            #print orig_peak_coord_np.shape
            
            dist_to_selected_peaks = dist.cdist(selected_peaks_coords_np,orig_peak_coord_np.reshape(1,3), 'euclidean')
        
            #print dist_to_selected_peaks
            
            min_dist_to_selected_peaks = np.amin(dist_to_selected_peaks,axis = 0)
            
            if min_dist < min_dist_to_selected_peaks:
                
                list_selected_peaks_coords.append(orig_peak_coord_np)
            
        
        else:
            list_selected_peaks_coords.append(orig_peak_coord)
            
        print len(list_selected_peaks_coords)
        
    return list_selected_peaks_coords
    
    
    
#def remove_close_peaks_neigh_in_binary_template_labels(list_orig_peak_coords,list_orig_peak_MNI_coords,template_data,template_labels,min_dist):
    
    
    #if len(list_orig_peak_coords) != len(list_orig_peak_MNI_coords):
        #print "!!!!!!!!!!!!!!!! Breaking !!!!!!!!!!!!!!!! list_orig_peak_coords %d and list_orig_peak_MNI_coords %d should have similar length" %(len(list_orig_peak_coords),len(list_orig_peak_MNI_coords))
        #return
    
    #img_shape = template_data.shape
    
    #indexed_mask_rois_data = np.zeros(img_shape,dtype = 'int64') -1
    
    #print indexed_mask_rois_data.shape
    
    #label_rois = []
    
    #list_selected_peaks_coords = []
    
    #list_rois_MNI_coords = []
    
    #orig_peak_coords_np = np.array(list_orig_peak_coords)
    
    #print type(orig_peak_coords_np),orig_peak_coords_np.dtype,orig_peak_coords_np.shape
    
    
    #orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)
           
    #print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
    ##for i,orig_peak_coord in enumerate([list_orig_peak_coords[0]]):
    #for i,orig_peak_coord in enumerate(list_orig_peak_coords):
    ##for orig_peak_coord in list_orig_peak_coords:
        
        #orig_peak_coord_np = np.array(orig_peak_coord)
            
        #if len(list_selected_peaks_coords) > 0:
            
            #selected_peaks_coords_np = np.array(list_selected_peaks_coords)
            
            ##orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)
            
            ##selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)
            
            
            ##print selected_peaks_coords_np.shape
            
            ##print orig_peak_coord_np.shape
            
            #dist_to_selected_peaks = dist.cdist(selected_peaks_coords_np,orig_peak_coord_np.reshape(1,3), 'euclidean')
        
            ##print dist_to_selected_peaks
            
            #min_dist_to_selected_peaks = np.amin(dist_to_selected_peaks,axis = 0)
            
            #if min_dist < min_dist_to_selected_peaks:
                        
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                ##list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
                #if peak_template_roi_index > 0:
                    
                    #neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
            
                    #indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
                
                    #label_rois.append(template_labels[peak_template_roi_index-1])
                    
                    ##print list_orig_peak_MNI_coords[i]
                    #list_rois_MNI_coords.append(list_orig_peak_MNI_coords[i])
                    
                    #list_selected_peaks_coords.append(orig_peak_coord_np)
                    
                    
            
        #else:
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
            ##list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
            #if peak_template_roi_index > 0:
                
                #neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
        
                #indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
            
                #label_rois.append(template_labels[peak_template_roi_index-1])
            
                ##print list_orig_peak_MNI_coords[i]
                #list_rois_MNI_coords.append(list_orig_peak_MNI_coords[i])
                
                #list_selected_peaks_coords.append(orig_peak_coord_np)
            
        #print len(list_selected_peaks_coords)
        
    #return list_selected_peaks_coords,indexed_mask_rois_data,label_rois,list_rois_MNI_coords
    
def remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords,template_data,min_dist):
    
    
    #if len(list_orig_peak_coords) != len(list_orig_peak_MNI_coords):
        #print "!!!!!!!!!!!!!!!! Breaking !!!!!!!!!!!!!!!! list_orig_peak_coords %d and list_orig_peak_MNI_coords %d should have similar length" %(len(list_orig_peak_coords),len(list_orig_peak_MNI_coords))
        #return
    
    img_shape = template_data.shape
    
    indexed_mask_rois_data = np.zeros(img_shape,dtype = 'int64') -1
    
    print indexed_mask_rois_data.shape
    
    list_selected_peaks_coords = []
    
    orig_peak_coords_np = np.array(list_orig_peak_coords)
    
    print type(orig_peak_coords_np),orig_peak_coords_np.dtype,orig_peak_coords_np.shape
    
    list_selected_peaks_indexes = []
    
    orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)
           
    print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
    #for i,orig_peak_coord in enumerate([list_orig_peak_coords[0]]):
    for i,orig_peak_coord in enumerate(list_orig_peak_coords):
        
        orig_peak_coord_np = np.array(orig_peak_coord)
            
        if len(list_selected_peaks_coords) > 0:
            
            selected_peaks_coords_np = np.array(list_selected_peaks_coords)
            
            #orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)
            
            #selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)
            
            
            #print selected_peaks_coords_np.shape
            
            #print orig_peak_coord_np.shape
            
            dist_to_selected_peaks = dist.cdist(selected_peaks_coords_np,orig_peak_coord_np.reshape(1,3), 'euclidean')
        
            #print dist_to_selected_peaks
            
            min_dist_to_selected_peaks = np.amin(dist_to_selected_peaks,axis = 0)
            
            #peak_template_roi_index = template_data[orig_peak_coord_np[0],orig_peak_coord_np[1],orig_peak_coord_np[2]]
            
            #print peak_template_roi_index
            
            #if peak_template_roi_index == 2:
            
                #print "Left Thalamus"
                #print orig_peak_coord_np
                
                #print min_dist_to_selected_peaks
                
                
                #0/0
                
            
            
            if min_dist < min_dist_to_selected_peaks:
                        
                list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
                if peak_template_roi_index > 0:
                    
                    neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
            
                    indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
                
                    list_selected_peaks_coords.append(orig_peak_coord_np)
                    
                    list_selected_peaks_indexes.append(i)
                    
                    print len(list_selected_peaks_coords)
                    
            
        else:
            list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
            if peak_template_roi_index > 0:
                
                neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
        
                indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
            
                list_selected_peaks_coords.append(orig_peak_coord_np)
                
                list_selected_peaks_indexes.append(i)
            
                print len(list_selected_peaks_coords)
        
    return list_selected_peaks_coords,indexed_mask_rois_data,list_selected_peaks_indexes
    
#def remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords,template_data,template_labels,min_dist = 3.0 * np.sqrt(3.0)):
    
    #img_shape = template_data.shape
    
    #indexed_mask_rois_data = np.zeros(img_shape,dtype = 'int64') -1
    
    #print indexed_mask_rois_data.shape
    
    #label_rois = []
    
    #list_selected_peaks_coords = []
    
    #orig_peak_coords_np = np.array(list_orig_peak_coords)
    
    #print type(orig_peak_coords_np),orig_peak_coords_np.dtype,orig_peak_coords_np.shape
    
    
    #orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)
           
    #print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
    #for orig_peak_coord in list_orig_peak_coords:
        
        #orig_peak_coord_np = np.array(orig_peak_coord)
            
        #if len(list_selected_peaks_coords) > 0:
            
            #selected_peaks_coords_np = np.array(list_selected_peaks_coords)
            
            ##orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)
            
            ##selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)
            
            
            ##print selected_peaks_coords_np.shape
            
            ##print orig_peak_coord_np.shape
            
            #dist_to_selected_peaks = dist.cdist(selected_peaks_coords_np,orig_peak_coord_np.reshape(1,3), 'euclidean')
        
            ##print dist_to_selected_peaks
            
            #min_dist_to_selected_peaks = np.amin(dist_to_selected_peaks,axis = 0)
            
            #if min_dist < min_dist_to_selected_peaks:
                        
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                ##list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
                #if peak_template_roi_index > 0:
                    
                    #neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
            
                    #indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
                
                    #label_rois.append(template_labels[peak_template_roi_index-1])
                    
                    #list_selected_peaks_coords.append(orig_peak_coord_np)
            
        #else:
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_binary_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
            ##list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
            #if peak_template_roi_index > 0:
                
                #neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
        
                #indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
            
                #label_rois.append(template_labels[peak_template_roi_index-1])
                
                #list_selected_peaks_coords.append(orig_peak_coord_np)
                
                
        #print len(list_selected_peaks_coords)
        
    #return list_selected_peaks_coords,indexed_mask_rois_data,label_rois
    
def remove_close_peaks_neigh_in_template(list_orig_peak_coords,template_data,template_labels,min_dist = 3.0 * np.sqrt(3)):
    
    img_shape = template_data.shape
    
    indexed_mask_rois_data = np.zeros(img_shape,dtype = 'int64') -1
    
    print indexed_mask_rois_data.shape
    
    label_rois = []
    
    list_selected_peaks_coords = []
    
    orig_peak_coords_np = np.array(list_orig_peak_coords)
    
    print type(orig_peak_coords_np),orig_peak_coords_np.dtype,orig_peak_coords_np.shape
    
    
    orig_peak_coords_dt = convert_np_coords_to_coords_dt(orig_peak_coords_np)
           
    print type(orig_peak_coords_dt),orig_peak_coords_dt.dtype,orig_peak_coords_dt.shape
    
    for orig_peak_coord in list_orig_peak_coords:
        
        orig_peak_coord_np = np.array(orig_peak_coord)
            
        if len(list_selected_peaks_coords) > 0:
            
            selected_peaks_coords_np = np.array(list_selected_peaks_coords)
            
            #orig_peak_coord_dt = convert_np_coords_to_coords_dt(orig_peak_coord)
            
            #selected_peaks_coords_dt = convert_np_coords_to_coords_dt(list_selected_peaks_coords)
            
            
            #print selected_peaks_coords_np.shape
            
            #print orig_peak_coord_np.shape
            
            dist_to_selected_peaks = dist.cdist(selected_peaks_coords_np,orig_peak_coord_np.reshape(1,3), 'euclidean')
        
            #print dist_to_selected_peaks
            
            min_dist_to_selected_peaks = np.amin(dist_to_selected_peaks,axis = 0)
            
            if min_dist < min_dist_to_selected_peaks:
                        
                list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
                if peak_template_roi_index > 0:
                    
                    neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
            
                    indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
                
                    label_rois.append(template_labels[peak_template_roi_index-1])
                    
                    list_selected_peaks_coords.append(orig_peak_coord_np)
            
        else:
            list_neigh_coords,peak_template_roi_index = return_indexed_mask_neigh_within_template(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
            #list_neigh_coords,peak_template_roi_index = return_indexed_mask_random_recursive_neigh_within_template_rois(orig_peak_coord_np,ROI_cube_size,template_data,orig_peak_coords_dt)
                
            if peak_template_roi_index > 0:
                
                neigh_coords = np.array(list_neigh_coords,dtype = 'int16')
        
                indexed_mask_rois_data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2]] = len(list_selected_peaks_coords)
            
                label_rois.append(template_labels[peak_template_roi_index-1])
                
                list_selected_peaks_coords.append(orig_peak_coord_np)
                
                
        print len(list_selected_peaks_coords)
        
    return list_selected_peaks_coords,indexed_mask_rois_data,label_rois
    

################################# preparing HO template by recombining sub and cortical mask + reslicing to image format ##################################
def compute_recombined_HO_template(img_header,img_affine,img_shape):

    HO_dir = "/usr/share/fsl/data/atlases/"
    
    ### cortical
    HO_cortl_img_file = os.path.join(HO_dir,"HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz")
    
    HO_cortl_img = nib.load(HO_cortl_img_file)
    
    HO_cortl_data = HO_cortl_img.get_data()
    
    print HO_cortl_data.shape
    
    print np.min(HO_cortl_data),np.max(HO_cortl_data)
    
    ### subcortical
    HO_sub_img_file = os.path.join(HO_dir,"HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
    
    HO_sub_img = nib.load(HO_sub_img_file)
    
    HO_sub_data = HO_sub_img.get_data()
    
    print HO_sub_data.shape
    
    print np.min(HO_sub_data),np.max(HO_sub_data)
    
    ################# White matter mask
    if not os.path.isfile(white_matter_HO_img_file):
        
        #### extracting mask for white matter :
        white_matter_HO_data = np.zeros(shape = HO_sub_data.shape,dtype = 'int')
        
        ###left white matter
        white_matter_HO_data[HO_sub_data == 1] = 1
        
        ### right white matter
        white_matter_HO_data[HO_sub_data == 12] = 1
        
        #### reslicing and saving white matter mask
        white_matter_HO_data = np.array(white_matter_HO_data,dtype = 'int')
        nib.save(nib.Nifti1Image(data = white_matter_HO_data,header = img_header,affine = img_affine),white_matter_HO_img_file)
            
        if not os.path.isfile(resliced_white_matter_HO_img_file):
            
            resliced_white_matter_HO_data = white_matter_HO_data[:img_shape[0],:img_shape[1],:img_shape[2]]
            
            resliced_white_matter_HO_data = white_matter_HO_data[6:-6,7:-7,10:-13]
            print resliced_white_matter_HO_data.shape
            
            nib.save(nib.Nifti1Image(data = resliced_white_matter_HO_data,header = img_header,affine = img_affine),resliced_white_matter_HO_img_file)
            
    ################# grey matter mask
    if not os.path.isfile(grey_matter_HO_img_file):
        
        #### extracting mask for grey matter :
        grey_matter_HO_data = np.zeros(shape = HO_sub_data.shape,dtype = 'int')
        
        ###left grey matter
        grey_matter_HO_data[HO_sub_data == 2] = 1
        
        ### right grey matter
        grey_matter_HO_data[HO_sub_data == 13] = 1
        
        #### reslicing and saving grey matter mask
        grey_matter_HO_data = np.array(grey_matter_HO_data,dtype = 'int')
        nib.save(nib.Nifti1Image(data = grey_matter_HO_data,header = img_header,affine = img_affine),grey_matter_HO_img_file)
            
        if not os.path.isfile(resliced_grey_matter_HO_img_file):
            
            resliced_grey_matter_HO_data = grey_matter_HO_data[:img_shape[0],:img_shape[1],:img_shape[2]]
            
            resliced_grey_matter_HO_data = grey_matter_HO_data[6:-6,7:-7,10:-13]
            print resliced_grey_matter_HO_data.shape
            
            nib.save(nib.Nifti1Image(data = resliced_grey_matter_HO_data,header = img_header,affine = img_affine),resliced_grey_matter_HO_img_file)
            
            
    ######## Ventricule (+ apprently outside the brain) mask
    if not os.path.isfile(ventricule_HO_img_file):
        
        #### extracting mask for ventricules:
        ventricule_HO_data = np.zeros(shape = HO_sub_data.shape,dtype = 'int')
        
        ### left ventricule
        ventricule_HO_data[HO_sub_data == 3] = 1
        
        ### right ventricule
        ventricule_HO_data[HO_sub_data == 14] = 1
        
        #### reslicing and saving white matter mask
        ventricule_HO_data = np.array(ventricule_HO_data,dtype = 'int')
        nib.save(nib.Nifti1Image(data = ventricule_HO_data,header = img_header,affine = img_affine),ventricule_HO_img_file)
        
    if not os.path.isfile(resliced_ventricule_HO_img_file):
        
        #### reslice SPM mask using HO target
        #reslice_ventricule_HO = spm.Reslice()
        #reslice_ventricule_HO.inputs.in_file = ventricule_HO_img_file
        #reslice_ventricule_HO.inputs.space_defining = spm_contrast_image_file
        
        #reslice_ventricule_HO.run()

        resliced_ventricule_HO_data = ventricule_HO_data[6:-6,7:-7,10:-13]
        print resliced_ventricule_HO_data.shape
        
        nib.save(nib.Nifti1Image(data = resliced_ventricule_HO_data,header = img_header,affine = img_affine),resliced_ventricule_HO_img_file)
        
    
    
    #useful_sub_indexes = [1] + range(3,11) + [12] + range(14,21)
    
    ### sans BrainStem
    useful_sub_indexes = [1] + range(3,7) + range(8,11) + [12] + range(14,21)
    print useful_sub_indexes
        
    useful_cortl_indexes = np.unique(HO_cortl_data)[:-1]
    
    print useful_cortl_indexes
    
    ####### concatenate areas from cortical and subcortical masks
    if not os.path.isfile(full_HO_img_file):
        
        ### recombining indexes
        full_HO_data = np.zeros(shape = HO_cortl_data.shape,dtype = 'int')
        
        #print full_HO_data
        
        
        new_index = 1
        
        for sub_index in useful_sub_indexes:
            
            print sub_index,new_index
            
            sub_mask = (HO_sub_data == sub_index + 1)
            
            print np.sum(sub_mask == 1)
            
            
            full_HO_data[sub_mask] = new_index
            
            new_index = new_index + 1
            
        for cortl_index in useful_cortl_indexes:
            
            print cortl_index,new_index
            
            cortl_mask = (HO_cortl_data == cortl_index + 1)
            
            print np.sum(cortl_mask == 1)
            
            
            full_HO_data[cortl_mask] = new_index
            
            new_index = new_index + 1
            
        full_HO_data = np.array(full_HO_data,dtype = 'int')
        
        print "Original HO template shape:"
        print full_HO_data.shape, np.min(full_HO_data),np.max(full_HO_data)
        
        nib.save(nib.Nifti1Image(data = full_HO_data,header = img_header,affine = img_affine),full_HO_img_file)
        
    print resliced_full_HO_img_file
    
    if not os.path.isfile(resliced_full_HO_img_file):
            
        resliced_full_HO_data = full_HO_data[6:-6,7:-7,10:-13]
        print resliced_full_HO_data.shape
        
        nib.save(nib.Nifti1Image(data = resliced_full_HO_data,header = img_header,affine = img_affine),resliced_full_HO_img_file)
        
    else: 
    
        
        ### loading results
        resliced_full_HO_img = nib.load(resliced_full_HO_img_file)
        
        resliced_full_HO_data = np.array(resliced_full_HO_img.get_data(),dtype = 'int')
        
        print "Resliced HO template shape:"
        print resliced_full_HO_data.shape
        
    
    #### reading HO labels and concatenate
    
    ### sub
    HO_sub_labels_file = os.path.join(HO_dir,"HarvardOxford-Subcortical.xml")
    
    xmldoc_sub = minidom.parse(HO_sub_labels_file)
    
    HO_sub_labels = [s.firstChild.data for i,s in enumerate(xmldoc_sub.getElementsByTagName('label')) if i in useful_sub_indexes]
    
    print HO_sub_labels
    
    ### cortl
    HO_cortl_labels_file = os.path.join(HO_dir,"HarvardOxford-Cortical-Lateralized.xml")
    
    xmldoc_cortl = minidom.parse(HO_cortl_labels_file)
    
    HO_cortl_labels = [s.firstChild.data for s in xmldoc_cortl.getElementsByTagName('label') if i in useful_cortl_indexes]
    
    print HO_cortl_labels
    
    HO_labels = HO_sub_labels + HO_cortl_labels
    
    print len(HO_labels)
    
    HO_abbrev_labels = []
    
    for label in HO_labels:
    
        abbrev_label = ""
        
        split_label_parts = label.split(",")
        
        #print len(split_label_parts)
        
        for i_part,label_part in enumerate(split_label_parts):
        
            split_label = label_part.split()
            
            if i_part == 0:
                
                #print split_label
                
                ### left right
                if len(split_label) > 0:
                    
                    if split_label[0] == "Left":
                    
                        abbrev_label = "L."
                        
                    elif split_label[0] == "Right":
                    
                        abbrev_label = "R."
                        
                    else:
                        #print split_label[0]
                        
                        abbrev_label = split_label[0].title() 
                        #continue
                        
                        
                if len(split_label) > 1:
                
                    #print split_label[1]
                    
                    abbrev_label = abbrev_label + split_label[1][:5].title() 
                    
                if len(split_label) > 2:
                
                    #print split_label[2]
                    
                    abbrev_label = abbrev_label + split_label[2][:3].title() 
                    
                #if len(split_label) > 3 and split_label[3] != "":
                    
                    #for i in range(3,len(split_label)):
                    
                        #if split_label[i] != "":
                        
                            #print i,split_label[i]
                            
                        
            if i_part == 1:
                    
                split_label = split_label_parts[1].split()
                
                #print split_label
                
                ### left right
                if len(split_label) > 0 and split_label[0] != "":
                    
                    abbrev_label = abbrev_label + "." + split_label[0][:4].title() 
                    
                    #print abbrev_label
                
                if len(split_label) > 1:
                
                    #print split_label[1]
                    
                    abbrev_label = abbrev_label + split_label[1][:3].title() 
                    #0/0

        print abbrev_label
        
        HO_abbrev_labels.append(abbrev_label)
        #0/0
            
    
    
    #0/0
    np_HO_abbrev_labels = np.array(HO_abbrev_labels,dtype = 'string')
    
    np_HO_labels = np.array(HO_labels,dtype = 'string')
    
    template_indexes = np.unique(resliced_full_HO_data)[1:]
    
    #print template_indexes
        
    print np_HO_labels.shape,np_HO_abbrev_labels.shape,template_indexes.shape
    
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_template = np.hstack((template_indexes.reshape(len(HO_labels),1),np_HO_labels.reshape(len(HO_labels),1),np_HO_abbrev_labels.reshape(len(HO_labels),1)))
    #,rois_MNI_coords))
    
    print info_template
   
    np.savetxt(info_template_file,info_template, fmt = '%s %s %s')
    
    return resliced_full_HO_data,np_HO_labels,np_HO_abbrev_labels
    
        
########################################### Activation peaks ROI template (computed once before the pipeline) ################################################


def compute_labelled_mask_from_HO_sub(resliced_full_HO_img_file,info_template_file,export_dir):

    labeled_mask = nib.load(resliced_full_HO_img_file)
    
    print labeled_mask
    
    labeled_mask_data = labeled_mask.get_data()
    
    labeled_mask_header = labeled_mask.get_header().copy()
    
    labeled_mask_affine = np.copy(labeled_mask.get_affine())
    
    
    useful_indexes = range(2,6) + range(7,10) + range(11,18)
    
    print useful_indexes
    
    ROI_mask_data = np.zeros(shape = labeled_mask_data.shape,dtype = labeled_mask_data.dtype)
    
    for label_index in useful_indexes:
        
        roi_mask = (labeled_mask_data == label_index)
        
        print np.sum(roi_mask == True)
        
        ROI_mask_data[roi_mask] = label_index
                
    print np.unique(ROI_mask_data)
    
    
    ROI_mask_file = os.path.join(export_dir,"ROI_mask.nii")
    
    nib.save(nib.Nifti1Image(np.array(ROI_mask_data,dtype = int),labeled_mask_affine,labeled_mask_header),ROI_mask_file)

    
    labels = [line.strip().split(' ')[-1] for line in open(info_template_file)]
    
    np_labels = np.array(labels,dtype = 'string')
    
    print labels
    
    useful_labels = np_labels[np.array(useful_indexes,dtype = 'int')-1]
    
    print useful_labels
    
    ROI_mask_labels_file = os.path.join(export_dir,"ROI_mask_labels.txt")
    
    np.savetxt(ROI_mask_labels_file,useful_labels,fmt = "%s")
    
    
    ##export each ROI in a single file (mask)
    for label_index in useful_indexes:
    
        roi_mask = (labeled_mask_data == label_index)
    
        single_ROI_mask = nib.Nifti1Image(np.array(roi_mask,dtype = int),labeled_mask_affine,labeled_mask_header)
        
        print np_labels[label_index - 1]
        
        single_ROI_mask_file = os.path.join(export_dir,"ROI_mask_HO_" + np_labels[label_index - 1] + ".nii")
        
        nib.save(single_ROI_mask,single_ROI_mask_file)
        
    return ROI_mask_file,ROI_mask_labels_file
    
def compute_labelled_mask_from_ROI_coords(neighbourhood = 1):
    """
    Compute labeled mask by specifying MNI coordinates and labels 'at hand'
    """
    from define_variables import resliced_full_HO_img_file,peak_activation_mask_analysis_name
    
    from define_variables import ROI_coords_mask_dir,ROI_coords_labelled_mask_file,ROI_coords_labels_file
    
    from labelled_mask import remove_close_peaks_neigh_in_binary_template
    
    from utils import check_np_dimension
    
    import itertools as iter
    
    from nipy.labs.viz import coord_transform

    import numpy as np
    import nibabel as nib
    
    orig_image = nib.load(resliced_full_HO_img_file)
    
    orig_image_data = orig_image.get_data()
    
    orig_image_data_shape = orig_image_data.shape
    
    print orig_image_data_shape
    
    orig_image_data_sform = orig_image.get_sform()
    
    print orig_image_data_sform
    
    ROI_MNI_coords_list = [[-30, 42,38],
                [42,28,-16],
                [0,-18,-16],
                [14,22,0],
                [-40,14,-14],
                [-42,54,14],
                [50,36,8],
                [44,52,-4],
                [-18,62,0],
                [-14,22,4],
                [-46,2,26],
                [8,34,22]]
                
    ROI_labels = ['MidSupFrontG',
                'PostOrbG',
                'VTA-SubNigria',
                'CaudHead',
                'Insula',
                'InfMidFrontG',
                'MidFrontG',
                'AntLatOrbG',
                'AntFrontG',
                'CaudHead',
                'InfFrontG',
                'AntCingG']
      
    print len(ROI_MNI_coords_list)
    print len(ROI_labels)
    
    ### transform MNI coords to numpy coords
    ## transfo inverse de celle stockes dans le header
    mni_sform_inv = np.linalg.inv(orig_image_data_sform)
    
    ROI_coords = np.array([coord_transform(x, y, z, mni_sform_inv) for x,y,z in ROI_MNI_coords_list],dtype = "int64")
    
    #print ROI_coords
    
    #ROI_coords_list = ROI_coords.tolist()
    
    #print ROI_coords_list
    
    #list_selected_peaks_coords,indexed_mask_rois_data,list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(ROI_coords_list,orig_image_data,min_dist_between_ROIs)
    
    #print list_selected_peaks_indexes
    #print len(list_selected_peaks_indexes)
    
    
    ROI_coords_labelled_mask = np.zeros(shape = orig_image_data_shape,dtype = 'int64') - 1
    
    print ROI_coords_labelled_mask
    
    
    neigh_range = range(-neighbourhood,neighbourhood+1)
    
    
    
    for i,ROI_coords in enumerate(ROI_coords):
    
        print ROI_coords
        
        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x,neigh_y,neigh_z = ROI_coords + relative_coord

            print neigh_x,neigh_y,neigh_z
            
            if check_np_dimension(ROI_coords_labelled_mask.shape,np.array([neigh_x,neigh_y,neigh_z],dtype = 'int64')):
            
                ROI_coords_labelled_mask[neigh_x,neigh_y,neigh_z] = i
            
            
    
    if not os.path.exists(ROI_coords_mask_dir):
        os.makedirs(ROI_coords_mask_dir)
        
    ###save ROI_coords_labelled_mask
    nib.save(nib.Nifti1Image(ROI_coords_labelled_mask,orig_image.get_affine(),orig_image.get_header()),ROI_coords_labelled_mask_file)
    
    ### save labels
    np.savetxt(ROI_coords_labels_file,ROI_labels,fmt = "%s")
    
    return ROI_coords_labelled_mask_file,ROI_coords_labels_file
    
#def compute_labelled_mask_from_spm_contrast_img():

    #### path and orig images
    #spm_contrast_path = os.path.join(nipype_analyses_path,l2_analysis_name_correlBehav,"level2_results_correlBehav_uncor_0_005/contrasts_thresh/_contrast_index_1_group_contrast_index_0")
    
    #spm_contrast_image_file = os.path.join(spm_contrast_path, 'spmT_0001_thr.img')

    #write_dir = os.path.join(nipype_analyses_path,cor_mat_analysis_name)
    
    #if not os.path.exists(write_dir):
        #os.makedirs(write_dir)

        
    
    ## prepare the data
    #img = load(spm_contrast_image_file)
    
    #img_header = img.get_header()
    #img_affine = img.get_affine()
    #img_shape = img.shape
    #img_data = img.get_data()
    
    ######## extract peaks from the contrast image
    
    #peaks =  stat_map.get_3d_peaks(image=img,mask=None, threshold = threshold,nn = cluster_nbvoxels)
    
    #print peaks[0]['ijk']
    #print len(peaks)
    
    #list_orig_peak_coords = [peak['ijk'] for peak in peaks]
    
    #print len(list_orig_peak_coords)
    
    #list_peak_positions = remove_close_peaks(list_orig_peak_coords,min_dist = ROI_cube_size * 2 * np.sqrt(3))
    
    #print len(list_peak_positions)
    
    #peak_coords = np.array(list_peak_positions)
    
    #print peak_coords.shape
    
    
    #print img_shape
    
    #indexed_mask_rois_data = np.zeros(img_shape,dtype = 'int64') -1
    
    #print indexed_mask_rois_data.shape
    
    #coord_rois = []
    
    #for index_roi in range(peak_coords.shape[0]):
    
        #roi_neigh_coords = return_indexed_mask(peak_coords[index_roi,],neighbourhood = ROI_cube_size)
        
        #indexed_mask_rois_data[roi_neigh_coords[:,0],roi_neigh_coords[:,1],roi_neigh_coords[:,2]] = index_roi
        
        #coord_rois.append([peak_coords[index_roi,0],peak_coords[index_roi,1],peak_coords[index_roi,2]])
        
    #print coord_rois
    
    #coord_rois = np.array(coord_rois,dtype = 'float')
    
    ##### exporting Rois image with different indexes 
    
    #indexed_rois_file =  os.path.join(write_dir, "rois_labelled_cluster_nbvoxels_" + str(cluster_nbvoxels) + "_size_" + str(ROI_cube_size) + ".nii")
    
    #nib.save(nib.Nifti1Image(data = indexed_mask_rois_data,header = img_header,affine = img_affine),indexed_rois_file)
    
    #### saving ROI coords as textfile
    #coord_rois_file =  os.path.join(write_dir, "coords_labelled_cluster_nbvoxels_" + str(cluster_nbvoxels) + "_size_" + str(ROI_cube_size) + ".txt")
    
    #np.savetxt(coord_rois_file,coord_rois, fmt = '%2.3f')
    
    #return indexed_rois_file,coord_rois_file

### labeled mask, contrained with HO template
def compute_labelled_mask_from_HO_and_spm_contrast_img():
    
    write_dir = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name)
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

        
    
    # prepare the data
    img = load(spm_contrast_image_file)
    
    img_header = img.get_header()
    img_affine = img.get_affine()
    img_shape = img.shape
    
    print img_shape
    
    img_data = img.get_data()
    
    
    ########################## Computing combined HO areas 
    
    resliced_full_HO_data,np_HO_labels,np_HO_abbrev_labels = compute_recombined_HO_template(img_header,img_affine,img_shape)
    
    #np.savetxt(info_template_file,info_rois, fmt = '%s %s %s %s %s %s')
      
      
      
    #### get peaks (avec la fonction stat_map.get_3d_peaks)
    
    peaks =  stat_map.get_3d_peaks(image=img,mask=None, threshold = threshold,nn = cluster_nbvoxels)
    
    print len(peaks)
    
    list_orig_peak_coords = [peak['ijk'] for peak in peaks]
    list_orig_peak_MNI_coords = [peak['pos'] for peak in peaks]
    
    
    print len(list_orig_peak_coords)
    
    #print list_orig_peak_MNI_coords
    #print len(list_orig_peak_MNI_coords)
        
    #### selectionne les pics sur leur distance entre eux et sur leur appatenance au template HO
    #list_selected_peaks_coords,indexed_mask_rois_data,label_rois = remove_close_peaks_neigh_in_template(list_orig_peak_coords,resliced_full_HO_data,HO_labels,min_dist = ROI_cube_size * 3 * np.sqrt(3.0))
    #list_selected_peaks_coords,indexed_mask_rois_data,label_rois = remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords,resliced_full_HO_data,HO_labels,min_dist = min_dist_between_ROIs)
    #list_selected_peaks_coords,indexed_mask_rois_data,label_rois,list_rois_MNI_coords = remove_close_peaks_neigh_in_binary_template_labels(list_orig_peak_coords,list_orig_peak_MNI_coords,resliced_full_HO_data,HO_abbrev_labels,min_dist_between_ROIs)
    list_selected_peaks_coords,indexed_mask_rois_data,list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords,resliced_full_HO_data,min_dist_between_ROIs)
    
    print list_selected_peaks_indexes
    print len(list_selected_peaks_indexes)
    
    #for coord in list_selected_peaks_coords:
    
        #print coord
    ##template_indexes = 
        #print resliced_full_HO_data[coord[0],coord[1],coord[2]]
    
    template_indexes = np.array([resliced_full_HO_data[coord[0],coord[1],coord[2]] for coord in list_selected_peaks_coords],dtype = 'int64')
    
    print template_indexes-1
    
    label_rois = np_HO_abbrev_labels[template_indexes-1]
    full_label_rois = np_HO_labels[template_indexes-1]
    
    #print label_rois2
    
    print label_rois
    
    #### exporting Rois image with different indexes 
    print np.unique(indexed_mask_rois_data)[1:].shape
    nib.save(nib.Nifti1Image(data = indexed_mask_rois_data,header = img_header,affine = img_affine),indexed_mask_rois_file)
    
    #### saving ROI coords as textfile
    np.savetxt(coord_rois_file,np.array(list_selected_peaks_coords,dtype = int), fmt = '%d')
    
    #### saving MNI coords as textfile
    list_rois_MNI_coords = [list_orig_peak_MNI_coords[index] for index in list_selected_peaks_indexes]
    
    print list_rois_MNI_coords
    
    rois_MNI_coords = np.array(list_rois_MNI_coords,dtype = int)
    np.savetxt(MNI_coord_rois_file,rois_MNI_coords, fmt = '%d')
    
    
    #### saving labels 
    np.savetxt(label_rois_file,label_rois, fmt = '%s')
    
    ### saving all together for infosource
    np_label_rois = np.array(label_rois,dtype = 'string').reshape(len(label_rois),1)
    np_full_label_rois = np.array(full_label_rois,dtype = 'string').reshape(len(full_label_rois),1)
    
    print np_label_rois.shape
    print rois_MNI_coords.shape
    
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    
    print info_rois
   
    np.savetxt(info_rois_file,info_rois, fmt = '%s %s %s %s %s %s')
    
    
    return indexed_mask_rois_file,coord_rois_file
    
def compute_labelled_mask_from_HO_and_merged_thr_spm_mask():
    
    
    write_dir = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name)
    
    print write_dir
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    spm_mask_files =  []
    
    print spm_contrasts_path
    
    for cont_index in contrast_indexes:
        for spm_index in spm_contrast_indexes:
            spm_mask_files.append(os.path.join(spm_contrasts_path,"_contrast_index_" + str(cont_index) + "_stim_OI/_l2Thresh_uncor_0_005_" + str(spm_index) + "/spmT_" + str(spm_index+1).zfill(4) + "_thr.img"))
        
    #spm_mask_files.sort()
    
    print spm_mask_files
    print len(spm_mask_files)
    
    # prepare the data
    img = nib.load(spm_mask_files[0])
    
    img_header = img.get_header()
    img_affine = img.get_affine()
    img_shape = img.shape
    
    img_data = img.get_data()
    
    ########################## Computing combined HO areas 
    
    resliced_full_HO_data,HO_labels,HO_abbrev_labels = compute_recombined_HO_template(img_header,img_affine,img_shape)
    
    ########################## Creating peak activation mask contrained by HO areas 
    
    #print len(HO_abbrev_labels)
    #print len(HO_labels)
    
    #0/0
    np_HO_abbrev_labels = np.array(HO_abbrev_labels,dtype = 'string')
    
    np_HO_labels = np.array(HO_labels,dtype = 'string')
    
    template_indexes = np.unique(resliced_full_HO_data)[1:]
    
    #print template_indexes
        
    print np_HO_labels.shape,np_HO_abbrev_labels.shape,template_indexes.shape
    
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_template = np.hstack((template_indexes.reshape(len(HO_labels),1),np_HO_labels.reshape(len(HO_labels),1),np_HO_abbrev_labels.reshape(len(HO_labels),1)))
    #,rois_MNI_coords))
    
    print info_template
   
    np.savetxt(info_template_file,info_template, fmt = '%s %s %s')
    
    merged_mask_data = np.zeros(shape = img_shape,dtype = float)
    
    print merged_mask_data.shape
    
    list_orig_ROI_spm_index = []
    
    ### list for all info about peaks after merging between different contrasts
    list_orig_peak_coords = []
    list_orig_peak_MNI_coords = []
    list_orig_peak_vals = []
    
    for i,spm_mask_file in enumerate(spm_mask_files):
        
        print spm_mask_file
        
        spm_mask_img = nib.load(spm_mask_file)
        
        spm_mask_data = spm_mask_img.get_data()
        
        #### get peaks (avec la fonction stat_map.get_3d_peaks)
        peaks =  stat_map.get_3d_peaks(image=spm_mask_img,mask=None)
        
        #print len(peaks)
        
        if peaks != None :
                
            print len(peaks)
            list_orig_peak_vals = list_orig_peak_vals + [peak['val'] for peak in peaks]
            list_orig_peak_coords = list_orig_peak_coords + [peak['ijk'] for peak in peaks]
            list_orig_peak_MNI_coords = list_orig_peak_MNI_coords + [peak['pos'] for peak in peaks]
        
            merged_mask_data[np.logical_and(spm_mask_data != 0.0, np.logical_not(np.isnan(spm_mask_data)))] += 2**i
            
            list_orig_ROI_spm_index = list_orig_ROI_spm_index +  [i+1] * len(peaks)
            
        print len(list_orig_peak_coords)
        print len(list_orig_ROI_spm_index)
    
    ### saving merged_mask
    nib.save(nib.Nifti1Image(data = merged_mask_data,header = img_header,affine = img_affine),merged_mask_img_file)
    
    #### selectionne les pics sur leur distance entre eux et sur leur appatenance au template HO

    list_selected_peaks_coords,indexed_mask_rois_data,list_selected_peaks_indexes = remove_close_peaks_neigh_in_binary_template(list_orig_peak_coords,resliced_full_HO_data,min_dist_between_ROIs)
    
    print list_selected_peaks_indexes
    print len(list_selected_peaks_indexes)
    
    template_indexes = np.array([resliced_full_HO_data[coord[0],coord[1],coord[2]] for coord in list_selected_peaks_coords],dtype = 'int64')
    print template_indexes
    
    np_HO_abbrev_labels = np.array(HO_abbrev_labels,dtype = 'string')
    
    np_HO_labels = np.array(HO_labels,dtype = 'string')
    
    print template_indexes-1
    
    label_rois = np_HO_abbrev_labels[template_indexes-1]
    full_label_rois = np_HO_labels[template_indexes-1]
    
    #print label_rois2
    
    print label_rois
    
    #### exporting Rois image with different indexes 
    nib.save(nib.Nifti1Image(data = indexed_mask_rois_data,header = img_header,affine = img_affine),indexed_mask_rois_file)
    
    #### saving ROI coords as textfile
    np.savetxt(coord_rois_file,np.array(list_selected_peaks_coords,dtype = int), fmt = '%d')
    
    #### saving MNI coords as textfile
    list_rois_MNI_coords = [list_orig_peak_MNI_coords[index] for index in list_selected_peaks_indexes]
    
    print list_rois_MNI_coords
    
    rois_MNI_coords = np.array(list_rois_MNI_coords,dtype = int)
    np.savetxt(MNI_coord_rois_file,rois_MNI_coords, fmt = '%d')
    
    ### orig index of peaks
    list_rois_orig_indexes = [list_orig_ROI_spm_index[index] for index in list_selected_peaks_indexes]
    
    print list_rois_orig_indexes
    
    rois_orig_indexes = np.array(list_rois_orig_indexes,dtype = int).reshape(len(list_rois_orig_indexes),1)
    
    print rois_orig_indexes.shape
    
    np.savetxt(rois_orig_indexes_file,rois_orig_indexes, fmt = '%d')
    
    
    
    #### mask with orig spm index
    orig_spm_index_mask_data = np.zeros(shape = img_shape,dtype = int)
    
    print np.unique(indexed_mask_rois_data)
    
    for i in np.unique(indexed_mask_rois_data)[1:]:
    
        print i,np.sum(indexed_mask_rois_data == i),rois_orig_indexes[i]
        
        orig_spm_index_mask_data[indexed_mask_rois_data == i] = rois_orig_indexes[i]
    
    nib.save(nib.Nifti1Image(data = orig_spm_index_mask_data,header = img_header,affine = img_affine),orig_spm_index_mask_file)
    
    #### saving labels 
    np.savetxt(label_rois_file,label_rois, fmt = '%s')
    
    ### saving all together for infosource
    np_label_rois = np.array(label_rois,dtype = 'string').reshape(len(label_rois),1)
    np_full_label_rois = np.array(full_label_rois,dtype = 'string').reshape(len(full_label_rois),1)
    
    print np_label_rois.shape
    print rois_MNI_coords.shape
    
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords))
    #info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),rois_MNI_coords))
    info_rois = np.hstack((np.unique(indexed_mask_rois_data)[1:].reshape(len(label_rois),1),np_full_label_rois,np_label_rois,rois_MNI_coords,rois_orig_indexes))
    
    print info_rois
   
    np.savetxt(info_rois_file,info_rois, fmt = '%s %s %s %s %s %s %s')
    
    
    return indexed_mask_rois_file,coord_rois_file
    
    
def compute_labelled_mask_from_HO_sub_jp():

    from define_variables_jp import resliced_full_HO_img_file,info_template_file,nipype_analyses_path,peak_activation_mask_analysis_name
    
    compute_labelled_mask_from_HO_sub(resliced_full_HO_img_file,info_template_file,export_dir = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name))
    
if __name__ =='__main__':
    
    ### compute labeled_mask from ROI coords
    #compute_labelled_mask_from_ROI_coords(ROI_dir = os.path.join(main_path,"D_extract_roi_beta_values"))
    
    ### compute labeled_mask from HO template
    #compute_labelled_mask_from_HO()
    
    ### export HO and single files 
    compute_labelled_mask_from_HO_sub_jp()
    
    ### compute labeled_mask from HO template + one contrast img
    #compute_labelled_mask_from_HO_and_spm_contrast_img()   
    
    ### compute labeled_mask from HO template + several spm thr mask
    #compute_labelled_mask_from_HO_and_merged_thr_spm_mask()
    
    
    
    
    
