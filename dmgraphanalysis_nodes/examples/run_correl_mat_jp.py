# -*- coding: utf-8 -*-
"""
A->B) First step: DICOM conversion
"""

#from nipype import config
#config.enable_debug_mode()

import sys, os
sys.path.append('../irm_analysis')


from  define_variables_jp import *
from dmgraphanalysis_nodes.nodes.correl_mat import ExtractTS,ExtractMeanTS,RegressCovar,FindSPMRegressor,ComputeConfCorMat

from dmgraphanalysis_nodes.labeled_mask import compute_labelled_mask_from_ROI_coords_files

############################################################## Workflow #############################################################
    
################################################ Infosource/Datasource

def create_inforsource_swra_jp():
    
    infosource = pe.Node(interface=IdentityInterface(fields=['subject_num', 'session','cond']),name="infosource")
    infosource.iterables = [('subject_num', behav_subject_ids_noA26),('session', funct_sessions_jp),('cond',condition_odors)]
    #infosource.iterables = [('subject_num', ['P06']),('session',['LIKING_OI']),('cond',['cheese'])]
    
    return infosource
    
def create_datasource_activation_jp():


    #### Data source from preprocessing
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_num','session'],outfields=['img_files','rp_files','spm_mat_file']),name = 'datasource')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    datasource.inputs.template = '%s/%s/%s%s%s%s/%s%s/%s%s*%s*%s'
    datasource.inputs.template_args = dict(
    img_files=[["merge_4d","","_session_",'session',"_subject_id_",'subject_num',"merge_funct","","swra",'subject_num',"",".nii"]],
    rp_files = [["SWRA_Files","","","","","subject_num","Run_",'session',"rp_",'subject_num','session',".txt"]],
    #rp_files = [["Files_ua",'subject_num',"FunctionalRuns_ua_SansImage1",'','',"rp_",'subject_num','',".txt"]],
    spm_mat_file = [[l1_analysis_name,"l1analysis","_session_",'session',"_subject_id_",'subject_num',"level1estimate","","SP","","",".mat"]]
    )

    datasource.inputs.sort_filelist = True
    
    return datasource
        
        
################################################# full analysis 
       
def create_correl_mat_by_session_workflow():
    
    main_workflow = Workflow(name=cor_mat_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### Info source
    #infosource = create_inforsource()
    infosource = create_inforsource_swra_jp()
    
    #### Data source from previous activation analysis
    #datasource = create_datasource_activation()
    datasource = create_datasource_activation_jp()
    
    main_workflow.connect(infosource, 'subject_num', datasource, 'subject_num')
    main_workflow.connect(infosource, 'session', datasource, 'session')
    
    ##### Preprocess pipeline
    
    #### extract mean time series for each ROI from labelled mask
    
    #extract_mean_ROI_ts = pe.MapNode(Function(input_names=['file_4D','indexed_rois_file'],output_names=['mean_masked_ts_file'],function=compute_mean_ts_from_labelled_mask),iterfield = ['in_file',funct_run_indexs,name='extract_mean_ROI_ts')
    #extract_mean_ROI_ts.inputs.indexed_rois_file = indexed_mask_rois_file
    
    #main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    
    
    
    ##### new version: use min_BOLD_intensity and return coords where signal is strong enough 
    #extract_mean_ROI_ts = pe.Node(interface = Function(input_names=['file_4D','indexed_rois_file','coord_rois_file'],output_names=['mean_masked_ts_file','subj_coord_rois_file'],function=compute_mean_ts_from_labelled_mask),name='extract_mean_ROI_ts')
    #extract_mean_ROI_ts.inputs.indexed_rois_file = indexed_mask_rois_file
    #extract_mean_ROI_ts.inputs.coord_rois_file = coord_rois_file
    
    #main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    
    
    
    #### Nodes version: use min_BOLD_intensity and return coords where signal is strong enough 
    extract_mean_ROI_ts = pe.Node(interface = ExtractTS(),name = 'extract_mean_ROI_ts')
    
    #(input_names=['file_4D','indexed_rois_file','coord_rois_file','min_BOLD_intensity'],output_names=['mean_masked_ts_file','subj_coord_rois_file'],function=compute_mean_ts_from_labelled_mask),name='extract_mean_ROI_ts')
    extract_mean_ROI_ts.inputs.indexed_rois_file = ROI_mask_file
    extract_mean_ROI_ts.inputs.coord_rois_file = ROI_coords_MNI_coords_file
    extract_mean_ROI_ts.inputs.min_BOLD_intensity = min_BOLD_intensity
    
    main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    
    #### extract white matter signal
    compute_wm_ts = pe.Node(interface = ExtractMeanTS(),iterfield = ['file_4D'],name = 'extract_wm_ts')
    
    compute_wm_ts.inputs.mask_file = resliced_white_matter_HO_img_file
    compute_wm_ts.inputs.suffix = 'wm'
    
    main_workflow.connect(datasource, ('img_files',show_files), compute_wm_ts, 'file_4D')
    
    #### extract csf signal
    compute_csf_ts = pe.Node(interface = ExtractMeanTS(),name = 'extract_csf_ts')
    
    compute_csf_ts.inputs.mask_file = resliced_ventricule_HO_img_file
    compute_csf_ts.inputs.suffix = 'csf'
    
    main_workflow.connect(datasource, ('img_files',show_files), compute_csf_ts, 'file_4D')
    
    
    
    #### regress covariates
    
    ### use R linear model to regress movement parameters, white matter and ventricule signals, and compute Z-score of the residuals
    #regress_covar = pe.MapNode(interface = RegressCovar(filtered = False, normalized = False),iterfield = ['masked_ts_file','rp_file','mean_wm_ts_file','mean_csf_ts_file'],name='regress_covar')
    regress_covar = pe.Node(interface = RegressCovar(),name='regress_covar')
    
    main_workflow.connect(extract_mean_ROI_ts, ('mean_masked_ts_file',show_files), regress_covar, 'masked_ts_file')
    main_workflow.connect(datasource, 'rp_files', regress_covar, 'rp_file')

    main_workflow.connect(compute_wm_ts, 'mean_masked_ts_file', regress_covar, 'mean_wm_ts_file')
    main_workflow.connect(compute_csf_ts, 'mean_masked_ts_file', regress_covar, 'mean_csf_ts_file')
    
    ### extract regressor of interest from SPM.mat
    extract_cond = pe.Node(interface = FindSPMRegressor(only_positive_values = True),name='extract_cond')
    
    main_workflow.connect(datasource, ('spm_mat_file',show_files), extract_cond, 'spm_mat_file')
    main_workflow.connect(infosource, 'cond', extract_cond, 'regressor_name')
        
    
    
    #################################### compute correlations ####################################################
    
    ########### confidence interval (new version, mais pas encore sure....)
    
    compute_conf_cor_mat = pe.Node(interface = ComputeConfCorMat(),name='compute_conf_cor_mat')
    
    compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob
    
    main_workflow.connect(regress_covar, ('resid_ts_file',show_length), compute_conf_cor_mat, 'ts_file')
    main_workflow.connect(extract_cond, ('regressor_file',show_length), compute_conf_cor_mat, 'weight_file')
    
    
    
    
    ### compute cor_mat
    ############# older version, with Z cor mat only 
    #compute_Z_cor_mat = pe.Node(Function(input_names=['resid_ts_file','regressor_file'],output_names=['Z_cor_mat_file'],function=compute_Z_correlation_matrix),name='compute_Z_cor_mat')
    
    #main_workflow.connect(regress_covar, ('resid_ts_file',show_length), compute_Z_cor_mat, 'resid_ts_file')
    #main_workflow.connect(extract_cond, ('regressor_file',show_length), compute_Z_cor_mat, 'regressor_file')
    
    ### histograms
    
    #plot_hist = pe.Node(Function(input_names=['Z_cor_mat_file'],output_names=['plot_hist_cor_mat_file','plot_heatmap_cor_mat_file'],function=plot_hist_Z_cor_mat),name='plot_hist')
    #main_workflow.connect(compute_Z_cor_mat, 'Z_cor_mat_file',plot_hist,'Z_cor_mat_file')
    
    ############## var from R packages, to long 
    ##compute_var_cor_mat = pe.Node(Function(input_names=['resid_ts_file','regressor_file'],output_names=['cor_mat_file','sderr_cor_mat_file','pval_cor_mat_file'],function=compute_var_correlation_matrix),name='compute_var_cor_mat')
    
    ##main_workflow.connect(merge_runs, 'resid_ts_file', compute_var_cor_mat, 'resid_ts_file')
    ##main_workflow.connect(merge_runs, 'regressor_file', compute_var_cor_mat, 'regressor_file')
    
    ##plot_hist_var = pe.Node(Function(input_names=['cor_mat_file','sderr_cor_mat_file','pval_cor_mat_file'],output_names=['plot_hist_cor_mat_file','plot_heatmap_cor_mat_file','plot_hist_sderr_cor_mat_file','plot_heatmap_sderr_cor_mat_file','plot_hist_pval_cor_mat_file','plot_heatmap_pval_cor_mat_file'],function=plot_hist_var_cor_mat),name='plot_hist_var')
    
    ##main_workflow.connect(compute_var_cor_mat, 'cor_mat_file',plot_hist_var,'cor_mat_file')
    ##main_workflow.connect(compute_var_cor_mat, 'sderr_cor_mat_file',plot_hist_var,'sderr_cor_mat_file')
    ##main_workflow.connect(compute_var_cor_mat, 'pval_cor_mat_file',plot_hist_var,'pval_cor_mat_file')
    
    ############ confidence interval (new version, mais pas encore sure....)
    
    #compute_conf_cor_mat = pe.Node(Function(input_names=['resid_ts_file','regressor_file','conf_interval_prob'],output_names=['cor_mat_file','Z_cor_mat_file','conf_cor_mat_file'],function=compute_conf_correlation_matrix),name='compute_conf_cor_mat')
    #compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob
    
    #main_workflow.connect(regress_covar, 'resid_ts_file', compute_conf_cor_mat, 'resid_ts_file')
    #main_workflow.connect(extract_cond, 'regressor_file', compute_conf_cor_mat, 'regressor_file')
    
    
    #plot_hist_conf = pe.Node(Function(input_names=['cor_mat_file','Z_cor_mat_file','conf_cor_mat_file'],output_names=['plot_hist_cor_mat_file','plot_heatmap_cor_mat_file','plot_hist_cor_mat_file','plot_heatmap_cor_mat_file','plot_hist_conf_cor_mat_file','plot_heatmap_conf_cor_mat_file'],function=plot_hist_conf_cor_mat),name='plot_hist_conf')
    
    #main_workflow.connect(compute_conf_cor_mat, 'cor_mat_file',plot_hist_conf,'cor_mat_file')
    #main_workflow.connect(compute_conf_cor_mat, 'Z_cor_mat_file',plot_hist_conf,'Z_cor_mat_file')
    #main_workflow.connect(compute_conf_cor_mat, 'conf_cor_mat_file',plot_hist_conf,'conf_cor_mat_file')
    
    
    return main_workflow

def gather_Z_correl_values():

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading labels"
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]
    
    np_labels = np.array(labels, dtype = 'str')
    
    print labels
       
    for cond in condition_odors:
        
        for sess in funct_sessions_jp:
        
            print sess
        
            all_Z_cor_values = []
    
            for subject_num in behav_subject_ids_noA26:
                
                print subject_num
                
                Z_cor_mat_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"_cond_" + cond + "_session_" + sess + "_subject_num_" + subject_num,"compute_conf_cor_mat","Z_cor_mat_resid_ts.npy" )
                    
                print Z_cor_mat_file
                
                Z_cor_mat = np.load(Z_cor_mat_file)
                
                print Z_cor_mat
                print Z_cor_mat.shape
                
                upper_tri_indexes = np.triu_indices(Z_cor_mat.shape[0],k=1)
                
                print upper_tri_indexes
                
                vect_Z_cor_mat = Z_cor_mat[upper_tri_indexes]
                
                print vect_Z_cor_mat
                
                all_Z_cor_values.append(vect_Z_cor_mat)
                
            np_all_Z_cor_values = np.array(all_Z_cor_values,dtype = 'f')
            
            print np_all_Z_cor_values.shape
            
            print np_labels[upper_tri_indexes[0]]
            
            labels_pairs = ["_".join((pair_lab[0],pair_lab[1])) for pair_lab in zip(np_labels[upper_tri_indexes[0]],np_labels[upper_tri_indexes[1]]) ]
            
            print labels_pairs
            
            
            
            df_all_Z_cor_values = pd.DataFrame(np_all_Z_cor_values,columns = labels_pairs,index = behav_subject_ids_noA26)
            
            df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values3_" + sess + '_' + cond +'.txt')
            #df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values_" + sess + '_' + cond +'.txt')
            #df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values2_" + sess + '_' + cond +'.txt')
            
            df_all_Z_cor_values.to_csv(df_all_Z_cor_values_file)
            
def gather_Z_correl_values_by_pair():

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading labels"
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]
    
    np_labels = np.array(labels, dtype = 'str')
    
    print labels
    
    upper_tri_indexes = np.triu_indices(np_labels.shape[0],k=1)
    
    print upper_tri_indexes
    
    print np_labels[upper_tri_indexes[0]]
    
    labels_pairs = ["_".join((pair_lab[0],pair_lab[1])) for pair_lab in zip(np_labels[upper_tri_indexes[0]],np_labels[upper_tri_indexes[1]]) ]
    
    print labels_pairs
    
    all_Z_cor_values = []
    
    sess_cond_names = []
    
    
    for sess in funct_sessions_jp:
    
        print sess
    
        sess_Z_cor_values = []
        for cond in condition_odors:
        
            print cond
            
            sess_cond_names.append(sess + '_' + cond)
            
            cond_Z_cor_values = []
             
            for subject_num in behav_subject_ids_noA26:
                
                print subject_num
                
                Z_cor_mat_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"_cond_" + cond + "_session_" + sess + "_subject_num_" + subject_num,"compute_conf_cor_mat","Z_cor_mat_resid_ts.npy" )
                    
                print Z_cor_mat_file
                
                Z_cor_mat = np.load(Z_cor_mat_file)
                
                print Z_cor_mat
                print Z_cor_mat.shape
                
                vect_Z_cor_mat = Z_cor_mat[upper_tri_indexes]
                
                print vect_Z_cor_mat
                
                cond_Z_cor_values.append(vect_Z_cor_mat)
                
            sess_Z_cor_values.append(cond_Z_cor_values)
        all_Z_cor_values.append(sess_Z_cor_values)
        
    np_all_Z_cor_values = np.array(all_Z_cor_values,dtype = 'f')
    
    print np_all_Z_cor_values.shape
    
    print sess_cond_names
    
    for i,label in enumerate(labels_pairs):
    
    
        print labels
        
        print np_all_Z_cor_values[1,0,:,i]
        
        pair_data = np.reshape(np_all_Z_cor_values[:,:,:,i],(-1,np_all_Z_cor_values.shape[2]))
        
        print pair_data[2,:]
        
        print pair_data.shape
        
            
        df_all_Z_cor_values = pd.DataFrame(np.transpose(pair_data),columns = sess_cond_names,index = behav_subject_ids_noA26)
        
        #df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values_" + label +'.txt')
        df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values3_" + label +'.txt')
        #df_all_Z_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"Z_cor_values2_" + label +'.txt')
        
        df_all_Z_cor_values.to_csv(df_all_Z_cor_values_file)
        
              
    
    
    
    print labels_pairs
    
    
def gather_correl_values():

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading labels"
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]
    
    np_labels = np.array(labels, dtype = 'str')
    
    print labels
       
    for cond in condition_odors:
        
        for sess in funct_sessions_jp:
        
            print sess
        
            all_cor_values = []
    
            for subject_num in behav_subject_ids_noA26:
                
                print subject_num
                
                cor_mat_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"_cond_" + cond + "_session_" + sess + "_subject_num_" + subject_num,"compute_conf_cor_mat","cor_mat_resid_ts.npy" )
                    
                print cor_mat_file
                
                cor_mat = np.load(cor_mat_file)
                
                print cor_mat
                print cor_mat.shape
                
                upper_tri_indexes = np.triu_indices(cor_mat.shape[0],k=1)
                
                print upper_tri_indexes
                
                vect_cor_mat = cor_mat[upper_tri_indexes]
                
                print vect_cor_mat
                
                all_cor_values.append(vect_cor_mat)
                
            np_all_cor_values = np.array(all_cor_values,dtype = 'f')
            
            print np_all_cor_values.shape
            
            print np_labels[upper_tri_indexes[0]]
            
            labels_pairs = ["_".join((pair_lab[0],pair_lab[1])) for pair_lab in zip(np_labels[upper_tri_indexes[0]],np_labels[upper_tri_indexes[1]]) ]
            
            print labels_pairs
            
            
            
            df_all_cor_values = pd.DataFrame(np_all_cor_values,columns = labels_pairs,index = behav_subject_ids_noA26)
            
            df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values3_" + sess + '_' + cond +'.txt')
            #df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values2_" + sess + '_' + cond +'.txt')
            #df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values_" + sess + '_' + cond +'.txt')
            
            df_all_cor_values.to_csv(df_all_cor_values_file)
            
def gather_correl_values_by_pair():

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading labels"
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]
    
    np_labels = np.array(labels, dtype = 'str')
    
    print labels
    
    upper_tri_indexes = np.triu_indices(np_labels.shape[0],k=1)
    
    print upper_tri_indexes
    
    print np_labels[upper_tri_indexes[0]]
    
    labels_pairs = ["_".join((pair_lab[0],pair_lab[1])) for pair_lab in zip(np_labels[upper_tri_indexes[0]],np_labels[upper_tri_indexes[1]]) ]
    
    print labels_pairs
    
    all_cor_values = []
    
    sess_cond_names = []
    
    
    for sess in funct_sessions_jp:
    
        print sess
    
        sess_cor_values = []
        for cond in condition_odors:
        
            print cond
            
            sess_cond_names.append(sess + '_' + cond)
            
            cond_cor_values = []
             
            for subject_num in behav_subject_ids_noA26:
                
                print subject_num
                
                cor_mat_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"_cond_" + cond + "_session_" + sess + "_subject_num_" + subject_num,"compute_conf_cor_mat","cor_mat_resid_ts.npy" )
                    
                print cor_mat_file
                
                cor_mat = np.load(cor_mat_file)
                
                print cor_mat
                print cor_mat.shape
                
                vect_cor_mat = cor_mat[upper_tri_indexes]
                
                print vect_cor_mat
                
                cond_cor_values.append(vect_cor_mat)
                
            sess_cor_values.append(cond_cor_values)
        all_cor_values.append(sess_cor_values)
        
    np_all_cor_values = np.array(all_cor_values,dtype = 'f')
    
    print np_all_cor_values.shape
    
    print sess_cond_names
    
    for i,label in enumerate(labels_pairs):
    
        pair_data = np.reshape(np_all_cor_values[:,:,:,i],(-1,np_all_cor_values.shape[2]))
        
        df_all_cor_values = pd.DataFrame(np.transpose(pair_data),columns = sess_cond_names,index = behav_subject_ids_noA26)
        df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values3_" + label +'.txt')
        #df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values2_" + label +'.txt')
        #df_all_cor_values_file = os.path.join(nipype_analyses_path,cor_mat_analysis_name,"cor_values_" + label +'.txt')
        df_all_cor_values.to_csv(df_all_cor_values_file)
        
    print labels_pairs
    
if __name__ =='__main__':
    
    #if not os.path.isfile(ROI_mask_file):
        
        ##compute_labelled_mask_from_HO()
        #### compute ROI mask HO()
        #compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file)
            
    ##### compute preprocessing for weighted correlation matrices
    #main_workflow = create_correl_mat_by_session_workflow()
    
    ###main_workflow.write_graph(cor_mat_analysis_name + '_graph.dot',graph2use='flat', format = 'svg')
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    ################################ Attention, il ne semble pas que ca marche avec le multiprocess - semble de venir de l'utilisation des MapNodes avec Rpy..... ############################
    ################################################ Ca bouffe toute la mémoire !!!!!!!!!!!!!!!!! ######################################################################
    ##################################################################### Utilisé le run sequentiel de preferences ############################
    
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    
    #main_workflow.run()
        
    
    #gather_correl_values()
    #gather_correl_values_by_pair()
    
    
    gather_Z_correl_values()
    gather_Z_correl_values_by_pair()
