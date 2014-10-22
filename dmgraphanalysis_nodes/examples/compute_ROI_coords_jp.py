# -*- coding: utf-8 -*-
"""
A->B) First step: DICOM conversion
"""

#from nipype import config
#config.enable_debug_mode()

#import sys,io,os

import sys, os
sys.path.append('../irm_analysis')

from  define_variables_jp import *
#from utils_dtype_coord import *

#from compute_peak_labelled_mask import compute_labelled_mask_from_HO,compute_labelled_mask_from_HO_and_merged_spm_mask
    

from dmgraphanalysis_nodes.labelled_mask import compute_labelled_mask_from_ROI_coords_files,merge_coord_and_label_files

import pyplotbrain as ppb
import pyqtgraph as pg
import numpy as np

    
def compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,MNI_coords_file,neighbourhood = 1):
    """
    Compute labeled mask by specifying MNI coordinates and labels 'at hand'
    #"""
    #from define_variables import resliced_full_HO_img_file,peak_activation_mask_analysis_name
    
    #from define_variables import ROI_coords_mask_dir,ROI_coords_labelled_mask_file,ROI_coords_labels_file
    
    #from labelled_mask import remove_close_peaks_neigh_in_binary_template
    
    from nipype.utils.filemanip import split_filename as split_f
    
    
    
    
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
    
    ROI_MNI_coords_list = np.array(np.loadtxt(MNI_coords_file),dtype = 'int').tolist()
    
    print ROI_MNI_coords_list
    
    #ROI_labels = [lign.strip() for lign in open(labels_file)]
    
    #print labels
    
    print len(ROI_MNI_coords_list)
    #print len(ROI_labels)
    
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
    
    
    
    for i,ROI_coord in enumerate(ROI_coords):
    
        print ROI_coord
        
        for relative_coord in iter.product(neigh_range, repeat=3):

            neigh_x,neigh_y,neigh_z = ROI_coord + relative_coord

            print neigh_x,neigh_y,neigh_z
            
            if check_np_dimension(ROI_coords_labelled_mask.shape,np.array([neigh_x,neigh_y,neigh_z],dtype = 'int64')):
            
                ROI_coords_labelled_mask[neigh_x,neigh_y,neigh_z] = i
            
           
        
    #path, fname, ext = '','',''
    path, fname, ext = split_f(MNI_coords_file)
    
    ROI_coords_labelled_mask_file = os.path.join(path,"All_labelled_ROI-neigh_"+str(neighbourhood)+".nii")
    
    ROI_coords_np_coords_file = os.path.join(path,"All_ROI_np_coords.txt")
    
    ###save ROI_coords_labelled_mask
    nib.save(nib.Nifti1Image(ROI_coords_labelled_mask,orig_image.get_affine(),orig_image.get_header()),ROI_coords_labelled_mask_file)
    
    #### save np coords
    np.savetxt(ROI_coords_np_coords_file,np.array(ROI_coords,dtype = int),fmt = "%d")
    
    
    return ROI_coords_labelled_mask_file
    
    
    
    
def pyplotbrain_display_ROI_coords(ROI_coords_MNI_coords_file,ROI_coords_orig_constrast_file):

    ROI_coords_MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'int')
    
    
    print ROI_coords_MNI_coords

    orig_constrast = np.array(np.loadtxt(ROI_coords_orig_constrast_file,dtype = 'string'))
    
    print orig_constrast
    
    codes = np.unique(orig_constrast)
    
    print codes
    
    app = pg.mkQApp()


    view = ppb.addView(with_config = True, background_color = (1,0,0))

    view.params['cortical_mesh'] =  'BrainMesh_ICBM152'

    view.plot_mesh()
    
    view.params['cortical_alpha'] =  0.5

    view.change_alpha_mesh()
    
    #view.params['cortical_mesh'] =  'BrainMesh_ICBM152'

    #colors = [ 
                    #(1,0,0,.8),
                    #(0,1,0,.8),
                    #(0,0,1,.8),
                    #(0,0,1,.8),
                    #(0,1,1,.8),
                    #(0,1,1,.8),
                    #]

    colors = [ 
                    (1,0,0,.8),
                    (1,0,0,.8),
                    (0,1,0,.8),
                    (0,1,0,.8),
                    ]

    #for i,code in enumerate(codes):
    #for i,code in enumerate(['E1','E2']):
    for i,code in enumerate(['R2E1','E1','R2E2','E2']):
    #for i,code in enumerate(['R1','R2','R2E2','R2E1','E1','E2']):
        
        print i,code,colors[i]
        
        selected_coords = ROI_coords_MNI_coords[orig_constrast == code,:]
        
        print selected_coords.shape
        
        view.add_node(selected_coords, color = colors[i], size = 4)

        #n = 30
        
        #node_coords = np.random.randn(n, 3)*20
        #connection_with = np.zeros((n,n))
        #connection_with[1,2] = 3
        #connection_with[4,7] = 5
        #connection_with[5,2] = 6
        #connection_with[8,5] = 4.5
        #connection_with[2,4] = 6
        #connection_with[8,2] = 6
        
        #view.add_edge(node_coords,connection_with,color = color)

    #view.to_file('test1.png')
    #view.to_file('test2.jpg')

    app.exec_()

def pyplotbrain_display_one_ROI_coords(ROI_coords_MNI_coords_file):

    ROI_coords_MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'int')
    
    
    print ROI_coords_MNI_coords

    app = pg.mkQApp()


    view = ppb.addView(with_config = True, background_color = (1,0,0))

    view.params['cortical_mesh'] =  'BrainMesh_ICBM152'

    view.plot_mesh()
    
    view.params['cortical_alpha'] =  0.5

    view.change_alpha_mesh()
    
    #view.params['cortical_mesh'] =  'BrainMesh_ICBM152'

    view.add_node(ROI_coords_MNI_coords, color = (1,0,0,.8), size = 4)

    app.exec_()

    
if __name__ =='__main__':
    
    #if not (os.path.isfile(ROI_coords_labelled_mask_file) or os.path.isfile(coord_rois_file)) :
        ##compute_labelled_mask_from_HO()
        #compute_labelled_mask_from_HO_and_merged_spm_mask()
        ### compute ROI mask HO()
            
    #print ROI_coords_labelled_mask_file,coord_rois_file

    #### ROI defined from a file
    ### merge coord and label files 
    #all_ROI_coords_file,all_ROI_labels_file = merge_coord_and_label_files(ROI_coords_dir)
    
    ### compute mask from MNI coords 
    compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file,neighbourhood = neighbourhood)
    
    #pyplotbrain_display_ROI_coords(ROI_coords_MNI_coords_file,ROI_coords_orig_constrast_file)
    
    #pyplotbrain_display_one_ROI_coords(os.path.join(nipype_analyses_path,"ReseauALS-conj_corPos-Rec",'Coord_NetworkRec.txt'))
    
    