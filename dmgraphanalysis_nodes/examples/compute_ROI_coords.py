# -*- coding: utf-8 -*-
"""
A->B) First step: DICOM conversion
"""

#from nipype import config
#config.enable_debug_mode()

#import sys,io,os

import sys, os
sys.path.append('../irm_analysis')

from  define_variables import *
#from utils_dtype_coord import *

from dmgraphanalysis_nodes.labeled_mask import compute_recombined_HO_template

#,compute_labelled_mask_from_HO_and_merged_spm_mask
    

from dmgraphanalysis.labelled_mask import compute_labelled_mask_from_ROI_coords_files
#,merge_coord_and_label_files

import pyplotbrain as ppb
import pyqtgraph as pg
import numpy as np

    
def pyplotbrain_display_ROI_coords(ROI_coords_MNI_coords_file,ROI_coords_orig_constrast_file,export_path,pref):

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

    #colors = [ 
                    #(1,0,0,.8),
                    #(1,0,0,.8),
                    #(0,1,0,.8),
                    #(0,1,0,.8),
                    #]
                    
    colors = [ 
                    (1,0,0,.8),
                    (0,1,0,.8),
                    #(0,0,1,.8),
                    #(0,0,1,.8),
                    #(0,1,1,.8),
                    #(0,1,1,.8),
                    ]
                    
    #for i,code in enumerate(codes):
    #for i,code in enumerate(['E1','E2']):
    #for i,code in enumerate(['R2E1','E1','R2E2','E2']):
    #for i,code in enumerate(['R1','R2','R2E2','R2E1','E1','E2']):
        
    for i,code in enumerate(['R1','R2']):
        
        print i,code,colors[i]
        
        selected_coords = ROI_coords_MNI_coords[orig_constrast == code,:]
        
        print selected_coords.shape
        
        view.add_node(selected_coords, color = colors[i], size = 4)

        
    view.resize(1200,800)
    
    ## from right
    view.glview.setCameraPosition(distance = 250, azimuth = 0,elevation = 0)
    view.to_file(os.path.join(export_path,pref + '_from_right.png'))
    
    ### from front
    view.glview.setCameraPosition(distance = 280,azimuth = 90,elevation = 0)
    view.to_file(os.path.join(export_path,pref + '_from_front.png'))

    ### from top
    view.glview.setCameraPosition(distance = 250,azimuth = 180,elevation = 90)
    view.to_file(os.path.join(export_path,pref + '_from_top.png'))

    ### from left
    view.glview.setCameraPosition(distance = 250,azimuth = 180,elevation = 0)
    view.to_file(os.path.join(export_path,pref + '_from_left.png'))

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
    
    #if not os.path.exists(resliced_full_HO_img_file)
    
        #resliced_full_HO_img_file = compute_labelled_mask_from_HO(ref_img)
            
            
    #if not os.path.exists(ROI_coords_labelled_mask_file):
        
        ### compute mask from MNI coords 
        #compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file,neighbourhood = neighbourhood)
        
        #compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file,neighbourhood = neighbourhood)
        
        #compute_labelled_mask_from_HO_and_merged_spm_mask()
        ### compute ROI mask HO()
            
    #print ROI_coords_labelled_mask_file,coord_rois_file

    #### ROI defined from a file
    ### merge coord and label files 
    #all_ROI_coords_file,all_ROI_labels_file = merge_coord_and_label_files(ROI_coords_dir)
    
    pyplotbrain_display_ROI_coords(ROI_coords_MNI_coords_file,ROI_coords_orig_constrast_file,export_path = ROI_coords_dir, pref = 'NodeCoords_R1R2')
    
    #pyplotbrain_display_one_ROI_coords(os.path.join(nipype_analyses_path,"ReseauALS-conj_corPos-Rec",'Coord_NetworkRec.txt'))
    
    