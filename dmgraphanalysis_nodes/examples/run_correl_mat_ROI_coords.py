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

#from dmgraphanalysis_nodes.correl_mat import ExtractTS,ExtractMeanTS,RegressCovar,FindSPMRegressor,MergeRuns,ComputeConfCorMat
from dmgraphanalysis_nodes.nodes.correl_mat import ExtractTS,ExtractMeanTS,RegressCovar,FindSPMRegressor,MergeRuns,ComputeConfCorMat

from dmgraphanalysis_nodes.utils import show_files,show_length

############################################################## Workflow #############################################################
    
################################################ Infosource/Datasource

def create_inforsource():
    
    infosource = pe.Node(interface=IdentityInterface(fields=['subject_num', 'cond']),name="infosource")
    
    ### all subjects in one go
    infosource.iterables = [('subject_num', subject_nums),('cond',epi_cond)]
    
    ## test
    #infosource.iterables = [('subject_num', ['S02']),('cond',epi_cond)]
    #infosource.iterables = [('subject_num', ['S02']),('cond',['Odor_Hit-WWW'])]
    
    return infosource
    
def create_datasource_activation_3D():

    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_num'],outfields=['img_files','rp_files','spm_mat_file']),name = 'datasource')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    datasource.inputs.template = '%s/%s/*%s*%s/%s/%s*%s*%s'
    datasource.inputs.template_args = dict(
    img_files=[[preproc_name,"preprocess_ua",'','subject_num',"normalize_ua_files_forth_WBEPI","wuaf",'',".nii"]],
    rp_files = [["Files_uaf",'subject_num',"Runs_",'','',"rp_",'subject_num',".txt"]],
    #rp_files = [["Files_ua",'subject_num',"FunctionalRuns_ua_SansImage1",'','',"rp_",'subject_num','',".txt"]],
    spm_mat_file = [[l1_analysis_name,"l1analysis",'subject_num','',"level1estimate","SP","",".mat"]]
    )

    datasource.inputs.sort_filelist = True
    
    return datasource
    
def create_datasource_activation():

    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_num'],outfields=['img_files','rp_files','spm_mat_file']),name = 'datasource')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    datasource.inputs.template = '%s/%s/*%s*%s/%s/%s*%s*%s*%s'
    datasource.inputs.template_args = dict(
    img_files=[[preproc_name,"preprocess_ua",'','subject_num',"normalize_ua_files_forth_WBEPI","wua4D",'subject_num','',".nii"]],
    rp_files = [["Files_ua",'subject_num',"FunctionalRuns_ua",'','',"rp_",'subject_num','',".txt"]],
    #rp_files = [["Files_ua",'subject_num',"FunctionalRuns_ua_SansImage1",'','',"rp_",'subject_num','',".txt"]],
    spm_mat_file = [[l1_analysis_name,"l1analysis",'subject_num','',"level1estimate","SP","","",".mat"]]
    )

    datasource.inputs.sort_filelist = True
    
    return datasource
        
        
################################################# full analysis 
       
def create_correl_mat_workflow():
    
    main_workflow = Workflow(name=cor_mat_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### Info source
    infosource = create_inforsource()
    
    #### Data source from previous activation analysis
    #datasource = create_datasource_activation()
    datasource = create_datasource_activation_3D()
    
    main_workflow.connect(infosource, 'subject_num', datasource, 'subject_num')
    
    ###### Preprocess pipeline
    
    #### extract mean time series for each ROI from labelled mask
    
    #extract_mean_ROI_ts = pe.MapNode(Function(input_names=['file_4D','indexed_rois_file'],output_names=['mean_masked_ts_file'],function=compute_mean_ts_from_labelled_mask),iterfield = ['in_file',funct_run_indexs,name='extract_mean_ROI_ts')
    #extract_mean_ROI_ts.inputs.indexed_rois_file = indexed_mask_rois_file
    
    #main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    
    
    
    ##### new version: use min_BOLD_intensity and return coords where signal is strong enough 
    #extract_mean_ROI_ts = pe.MapNode(interface = Function(input_names=['file_4D','indexed_rois_file','coord_rois_file','min_BOLD_intensity'],output_names=['mean_masked_ts_file','subj_coord_rois_file'],function=compute_mean_ts_from_labelled_mask),iterfield = ['file_4D'],name='extract_mean_ROI_ts')
    #extract_mean_ROI_ts.inputs.indexed_rois_file = ROI_coords_labelled_mask_file
    #extract_mean_ROI_ts.inputs.coord_rois_file = ROI_coords_file
    #extract_mean_ROI_ts.inputs.min_BOLD_intensity = min_BOLD_intensity
    
    #main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    #### Nodes version: use min_BOLD_intensity and return coords where signal is strong enough 
    extract_mean_ROI_ts = pe.MapNode(interface = ExtractTS(),iterfield = ['file_4D'],name = 'extract_mean_ROI_ts')
    
    #(input_names=['file_4D','indexed_rois_file','coord_rois_file','min_BOLD_intensity'],output_names=['mean_masked_ts_file','subj_coord_rois_file'],function=compute_mean_ts_from_labelled_mask),name='extract_mean_ROI_ts')
    extract_mean_ROI_ts.inputs.indexed_rois_file = ROI_coords_labelled_mask_file
    extract_mean_ROI_ts.inputs.coord_rois_file = ROI_coords_MNI_coords_file
    extract_mean_ROI_ts.inputs.min_BOLD_intensity = min_BOLD_intensity
    
    main_workflow.connect(datasource, ('img_files',show_files), extract_mean_ROI_ts, 'file_4D')
    
    
    #### extract white matter signal
    compute_wm_ts = pe.MapNode(interface = ExtractMeanTS(),iterfield = ['file_4D'],name = 'extract_wm_ts')
    
    compute_wm_ts.inputs.mask_file = resliced_white_matter_HO_img_file
    compute_wm_ts.inputs.suffix = 'wm'
    
    main_workflow.connect(datasource, ('img_files',show_files), compute_wm_ts, 'file_4D')
    
    #### extract csf signal
    compute_csf_ts = pe.MapNode(interface = ExtractMeanTS(),iterfield = ['file_4D'],name = 'extract_csf_ts')
    
    compute_csf_ts.inputs.mask_file = resliced_ventricule_HO_img_file
    compute_csf_ts.inputs.suffix = 'csf'
    
    main_workflow.connect(datasource, ('img_files',show_files), compute_csf_ts, 'file_4D')
    
    
    #### regress covariates
    
    ### use R linear model to regress movement parameters, white matter and ventricule signals, and compute Z-score of the residuals
    #regress_covar = pe.MapNode(interface = RegressCovar(filtered = False, normalized = False),iterfield = ['masked_ts_file','rp_file','mean_wm_ts_file','mean_csf_ts_file'],name='regress_covar')
    regress_covar = pe.MapNode(interface = RegressCovar(),iterfield = ['masked_ts_file','rp_file','mean_wm_ts_file','mean_csf_ts_file'],name='regress_covar')
    
    main_workflow.connect(extract_mean_ROI_ts, ('mean_masked_ts_file',show_files), regress_covar, 'masked_ts_file')
    main_workflow.connect(datasource, 'rp_files', regress_covar, 'rp_file')

    main_workflow.connect(compute_wm_ts, 'mean_masked_ts_file', regress_covar, 'mean_wm_ts_file')
    main_workflow.connect(compute_csf_ts, 'mean_masked_ts_file', regress_covar, 'mean_csf_ts_file')
    
    
    
    
    ### extract regressor of interest from SPM.mat
    extract_cond = pe.MapNode(interface = FindSPMRegressor(only_positive_values = True),iterfield = ['run_index'],name='extract_cond')
    
    main_workflow.connect(datasource, ('spm_mat_file',show_files), extract_cond, 'spm_mat_file')
    main_workflow.connect(infosource, 'cond', extract_cond, 'regressor_name')
    
    extract_cond.inputs.run_index = funct_run_indexs
    
    
    ### merge_runs new version: merge also coords (if different between sessions)
    merge_runs = pe.Node(interface = MergeRuns(),name='merge_runs')
    
    main_workflow.connect(extract_mean_ROI_ts, ('subj_coord_rois_file',show_length), merge_runs, 'coord_rois_files')
    main_workflow.connect(regress_covar, ('resid_ts_file',show_length), merge_runs, 'ts_files')
    main_workflow.connect(extract_cond, ('regressor_file',show_length), merge_runs, 'regressor_files')
    
    #################################### compute correlations ####################################################
    
    ## compute cor_mat
    ############ older version, with Z cor mat only 
    #compute_Z_cor_mat = pe.Node(Function(input_names=['resid_ts_file','regressor_file'],output_names=['Z_cor_mat_file'],function=compute_Z_correlation_matrix),name='compute_Z_cor_mat')
    
    #main_workflow.connect(merge_runs, 'resid_ts_file', compute_Z_cor_mat, 'resid_ts_file')
    #main_workflow.connect(merge_runs, 'regressor_file', compute_Z_cor_mat, 'regressor_file')
    
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
    
    ########### confidence interval (new version, mais pas encore sure....)
    
    compute_conf_cor_mat = pe.Node(interface = ComputeConfCorMat(),name='compute_conf_cor_mat')
    
    compute_conf_cor_mat.inputs.conf_interval_prob = conf_interval_prob
    
    main_workflow.connect(merge_runs, 'ts_all_runs_file', compute_conf_cor_mat, 'ts_file')
    main_workflow.connect(merge_runs, 'regressor_all_runs_file', compute_conf_cor_mat, 'weight_file')
    
    
    #plot_hist_conf = pe.Node(Function(input_names=['cor_mat_file','Z_cor_mat_file','conf_cor_mat_file'],output_names=['plot_hist_cor_mat_file','plot_heatmap_cor_mat_file','plot_hist_cor_mat_file','plot_heatmap_cor_mat_file','plot_hist_conf_cor_mat_file','plot_heatmap_conf_cor_mat_file'],function=plot_hist_conf_cor_mat),name='plot_hist_conf')
    
    #main_workflow.connect(compute_conf_cor_mat, 'cor_mat_file',plot_hist_conf,'cor_mat_file')
    #main_workflow.connect(compute_conf_cor_mat, 'Z_cor_mat_file',plot_hist_conf,'Z_cor_mat_file')
    #main_workflow.connect(compute_conf_cor_mat, 'conf_cor_mat_file',plot_hist_conf,'conf_cor_mat_file')
    
    return main_workflow
    
if __name__ =='__main__':
    
    #if not (os.path.isfile(indexed_mask_rois_file) or os.path.isfile(coord_rois_file)) :
        ##compute_labelled_mask_from_HO()
        #compute_labelled_mask_from_HO_and_merged_spm_mask()
        ### compute ROI mask HO()
            
    #print indexed_mask_rois_file,coord_rois_file

    #### compute preprocessing for weighted correlation matrices
    main_workflow = create_correl_mat_workflow()

    ##main_workflow.write_graph(cor_mat_analysis_name + '_graph.dot',graph2use='flat', format = 'svg')
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}

    ############################### Attention, il ne semble pas que ca marche avec le multiprocess - semble de venir de l'utilisation des MapNodes avec Rpy..... ############################
    ############################################### Ca bouffe toute la mémoire !!!!!!!!!!!!!!!!! ######################################################################
    #################################################################### Utilisé le run sequentiel de preferences ############################

    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})

    main_workflow.run()
        
    
    
    
