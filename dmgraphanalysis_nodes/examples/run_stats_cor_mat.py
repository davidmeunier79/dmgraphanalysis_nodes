# -*- coding: utf-8 -*-
"""
7th step: Compute similarity between pairs of partitions obtained after modular partitions
"""

import sys, os
sys.path.append('../irm_analysis')

from define_variables import *

from dmgraphanalysis.modularity import prep_radatools,community_radatools,plot_igraph_modules_conf_cor_mat_rada,export_lol_mask_file
from dmgraphanalysis.graph_stats import *

    ################################### datasource #######################################

##### By event conditions (Hit_WWW vs. Hit_What) #####################################################
    
def create_datasource_conf_cor_mat_by_cond_event():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'], outfields=['cor_mat_files1','coords_files1','cor_mat_files2','coords_files2']),name = 'datasource')
    datasource.inputs.sort_filelist = True
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    cor_mat_files1=[[cor_mat_analysis_name,'cond',"Hit-WWW","compute_conf_cor_mat","Z_cor_mat.npy"]],
    coords_files1= [[cor_mat_analysis_name,'cond',"Hit-WWW","merge_runs","coord_rois_all_runs.txt"]],
    
    
    cor_mat_files2=[[cor_mat_analysis_name,'cond',"Hit-What","compute_conf_cor_mat","Z_cor_mat.npy"]],
    coords_files2= [[cor_mat_analysis_name,'cond',"Hit-What","merge_runs","coord_rois_all_runs.txt"]]
    
    ) 
    
    
    return datasource
    
    
def run_rada_coclass_pairwise_NBS():
    
    #pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    
    main_workflow = Workflow(name= stat_cor_mat_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    if model_name == 'WWW_What_model8':
        
        #### Info source
        infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
        
        ### all 
        #infosource.iterables = [('cond', ['Odor','Rec','Recall'])]
        infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor','Rec','Recall'])]
        #infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor_preparation','Odor','Rec','Recall'])]
        
        ### test
        #infosource.iterables = [('cond', ['Odor'])]
        
        
        #### Data source
        datasource = create_datasource_conf_cor_mat_by_cond_event()
            
        main_workflow.connect(infosource, 'cond', datasource, 'cond')
        
        
    elif model_name == 'Model9_Only3events_WWW_What':
    
        
        #### Info source
        infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
        
        ### all 
        infosource.iterables = [('cond', ['Odor','Rec','Recall'])]
        #infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor','Rec','Recall'])]
        #infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor_preparation','Odor','Rec','Recall'])]
        
        ### test
        #infosource.iterables = [('cond', ['Odor'])]
        
        
        #### Data source
        datasource = create_datasource_conf_cor_mat_by_cond_event()
            
        main_workflow.connect(infosource, 'cond', datasource, 'cond')
        
    elif model_name == 'Hit_model8':
        
        #### Data source
        datasource = create_datasource_conf_cor_mat_by_cond_memory()
            
        
        
    ###################################### compute sum coclass and group based coclass matrices  #####################################################
    
    #### prepare_nbs_stats_rada
    
    prepare_cormat1 = pe.Node(Function(input_names=['cor_mat_files','coords_files','gm_mask_coords_file'],output_names = ['group_cormat_matrix_file','avg_cor_mat_matrix_file','group_vect_file'],function = prepare_nbs_stats_cor_mat),name='prepare_cormat1')
    prepare_cormat1.inputs.gm_mask_coords_file = coord_rois_file
    #prepare_cormat1.inputs.gm_mask_file = os.path.join(main_path,dartel_analysis_name ,"gm_mask","gm_mask.nii")
    
    main_workflow.connect(datasource, 'cor_mat_files1',prepare_cormat1,'cor_mat_files')
    main_workflow.connect(datasource, 'coords_files1',prepare_cormat1,'coords_files')
    
    
    prepare_cormat2 = pe.Node(Function(input_names=['cor_mat_files','coords_files','gm_mask_coords_file'],output_names = ['group_cormat_matrix_file','avg_cor_mat_matrix_file','group_vect_file'],function = prepare_nbs_stats_cor_mat),name='prepare_cormat2')
    prepare_cormat2.inputs.gm_mask_coords_file = coord_rois_file
    #prepare_cormat2.inputs.gm_mask_file = os.path.join(main_path,dartel_analysis_name ,"gm_mask","gm_mask.nii")
    
    main_workflow.connect(datasource, 'cor_mat_files2',prepare_cormat2,'cor_mat_files')
    main_workflow.connect(datasource, 'coords_files2',prepare_cormat2,'coords_files')
    
    ############################################################# nbs stats matrix #################################################################
    
    #nbs_coclass = pe.Node(Function(input_names=['group_coclass_matrix_file1','group_coclass_matrix_file2'],output_names = ['nbs_coclass_stats_mat_file','nbs_adj_mat_file'],function = compute_coclass_rada_nbs_stats),name='nbs_coclass_THRES_'+str(THRESH).replace('.','_')+'_K_'+str(K))
    
    #main_workflow.connect(prepare_coclass1, 'group_coclass_matrix_file',nbs_coclass,'group_coclass_matrix_file1')
    #main_workflow.connect(prepare_coclass2, 'group_coclass_matrix_file',nbs_coclass,'group_coclass_matrix_file2')
    
    ###### plot adj mat
    #plot_adj = pe.Node(Function(input_names=['nbs_adj_mat_file','gm_mask_coords_file','gm_mask_file'],output_names = ['plot_3D_nbs_adj_mat_file','heatmap_nbs_adj_mat_file','signif_degree_img_file'],function = plot_signif_nbs_adj_mat),name='plot_adj_THRES_'+str(THRESH).replace('.','_')+'_K_'+str(K))
    #plot_adj.inputs.gm_mask_coords_file = coord_rois_file
    #plot_adj.inputs.gm_mask_file = os.path.join(main_path,dartel_analysis_name ,"gm_mask","gm_mask.nii")
    
    #main_workflow.connect(nbs_coclass,'nbs_adj_mat_file',plot_adj,'nbs_adj_mat_file')
    
    ############################################################### pairwise stats ##################################################################
    
    ########## pairwise stats FDR (t-test)
    pairwise_stats_fdr = pe.Node(Function(input_names=['group_cormat_matrix_file1','group_cormat_matrix_file2','t_test_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_ttest_stats_fdr),name='pairwise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_stats_fdr.inputs.t_test_thresh_fdr = t_test_thresh_fdr
    
    main_workflow.connect(prepare_cormat1, 'group_cormat_matrix_file',pairwise_stats_fdr,'group_cormat_matrix_file1')
    main_workflow.connect(prepare_cormat2, 'group_cormat_matrix_file',pairwise_stats_fdr,'group_cormat_matrix_file2')
    
    
    plot_pairwise_stats = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels),name='plot_pairwise_stats')
    plot_pairwise_stats.inputs.coords_file = coord_rois_file
    plot_pairwise_stats.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats,'signed_bin_mat_file')
    
    plot_pairwise_stats_fdr = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_stats_fdr.inputs.coords_file = coord_rois_file
    plot_pairwise_stats_fdr.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats_fdr,'signed_bin_mat_file')
    
    #plot_pairwise_stats_fdr = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat),name='plot_pairwise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    #plot_pairwise_stats_fdr.inputs.coords_file = coord_rois_file
    
    #main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats_fdr,'signed_bin_mat_file')
    
    
    ########### pairwise stats (binomial test)
    #pairwise_stats = pe.Node(Function(input_names=['group_coclass_matrix_file1','group_coclass_matrix_file2'],output_names = ['pairwise_binom_adj_mat_file','degree_pairwise_binom_adj_mat_file'],function = compute_coclass_pairwise_binom_stats),name='pairwise_pairwise_stats_'+ str(conf_interval_binom).replace('.','_'))
    
    #main_workflow.connect(prepare_coclass1, 'group_coclass_matrix_file',pairwise_stats,'group_coclass_matrix_file1')
    #main_workflow.connect(prepare_coclass2, 'group_coclass_matrix_file',pairwise_stats,'group_coclass_matrix_file2')
    
    
    #plot_pairwise_stats = pe.Node(Function(input_names=['bin_mat_file','coords_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_bin_mat),name='plot_pairwise_stats_'+ str(conf_interval_binom).replace('.','_'))
    #plot_pairwise_stats.inputs.coords_file = coord_rois_file
    
    #main_workflow.connect(pairwise_stats,'pairwise_binom_adj_mat_file',plot_pairwise_stats,'bin_mat_file')
    
    #plot_degree_pairwise_stats = pe.Node(Function(input_names=['val_vect_file','indexed_mask_file'],output_names = ['val_vect_img_file'],function = plot_img_val_vect),name='plot_degree_pairwise_stats_'+ str(conf_interval_binom).replace('.','_'))
    #plot_degree_pairwise_stats.inputs.indexed_mask_file = indexed_mask_rois_file
    
    #main_workflow.connect(pairwise_stats,'degree_pairwise_binom_adj_mat_file',plot_degree_pairwise_stats,'val_vect_file')
    
    ############################################################### nodewise stats ####################################################################
    
    ############# nodewise stats (t-test) 
    #nodewise_stats = pe.Node(Function(input_names=['group_vect_file1','group_vect_file2'],output_names = ['nodewise_t_val_vect_file'],function = compute_coclass_nodewise_ttest_stats),name='nodewise_stats_'+ str(THRESH).replace('.','_'))
    
    #main_workflow.connect(prepare_coclass1, 'group_vect_file',nodewise_stats,'group_vect_file1')
    #main_workflow.connect(prepare_coclass2, 'group_vect_file',nodewise_stats,'group_vect_file2')
    
    
    #plot_nodewise_stats = pe.Node(Function(input_names=['val_vect_file','indexed_mask_file'],output_names = ['val_vect_img_file'],function = plot_img_val_vect),name='plot_nodewise_stats_'+ str(THRESH).replace('.','_'))
    #plot_nodewise_stats.inputs.indexed_mask_file = indexed_mask_rois_file
    
    #main_workflow.connect(nodewise_stats,'nodewise_t_val_vect_file',plot_nodewise_stats,'val_vect_file')
    
    
    ############ nodewise stats fdr (t-test) 
    nodewise_stats_fdr = pe.Node(Function(input_names=['group_vect_file1','group_vect_file2','t_test_thresh_fdr'],output_names = ['nodewise_t_val_vect_file'],function = compute_nodewise_ttest_stats_fdr),name='nodewise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    nodewise_stats_fdr.inputs.t_test_thresh_fdr = t_test_thresh_fdr
    
    main_workflow.connect(prepare_cormat1, 'group_vect_file',nodewise_stats_fdr,'group_vect_file1')
    main_workflow.connect(prepare_cormat2, 'group_vect_file',nodewise_stats_fdr,'group_vect_file2')
    
    
    plot_nodewise_stats_fdr = pe.Node(Function(input_names=['val_vect_file','indexed_mask_file'],output_names = ['val_vect_img_file'],function = plot_img_val_vect),name='plot_nodewise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_nodewise_stats_fdr.inputs.indexed_mask_file = indexed_mask_rois_file
    
    main_workflow.connect(nodewise_stats_fdr,'nodewise_t_val_vect_file',plot_nodewise_stats_fdr,'val_vect_file')
    
    
    ############################################################### pairwise correl with behav_score ###################################################
    
    ##### for group 1 (cond = WWW)
    
    ########## pairwise stats FDR (t-test) with scores_total
    pairwise_correl_scores_total_fdr1 = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_total_fdr1_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_total_fdr1.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_total_fdr1.inputs.behav_score = scores_total
    
    main_workflow.connect(prepare_cormat1, 'group_cormat_matrix_file',pairwise_correl_scores_total_fdr1,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_total_fdr1 = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_total_fdr1_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_total_fdr1.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_total_fdr1.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_total_fdr1,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_total_fdr1,'signed_bin_mat_file')
    
    
    
    ########## pairwise stats FDR (t-test) with scores_epi
    pairwise_correl_scores_epi_fdr1 = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_epi_fdr1_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_epi_fdr1.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_epi_fdr1.inputs.behav_score = scores_epi
    
    main_workflow.connect(prepare_cormat1, 'group_cormat_matrix_file',pairwise_correl_scores_epi_fdr1,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_epi_fdr1 = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_epi_fdr1_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_epi_fdr1.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_epi_fdr1.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_epi_fdr1,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_epi_fdr1,'signed_bin_mat_file')
    
    
    ##### for group 2 (cond = What)
    
    ########## pairwise stats FDR (t-test) with scores_total
    pairwise_correl_scores_total_fdr2 = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_total_fdr2_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_total_fdr2.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_total_fdr2.inputs.behav_score = scores_total
    
    main_workflow.connect(prepare_cormat2, 'group_cormat_matrix_file',pairwise_correl_scores_total_fdr2,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_total_fdr2 = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_total_fdr2_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_total_fdr2.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_total_fdr2.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_total_fdr2,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_total_fdr2,'signed_bin_mat_file')
    
    
    
    ########## pairwise stats FDR (t-test) with scores_epi
    pairwise_correl_scores_epi_fdr2 = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_epi_fdr2_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_epi_fdr2.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_epi_fdr2.inputs.behav_score = scores_epi
    
    main_workflow.connect(prepare_cormat2, 'group_cormat_matrix_file',pairwise_correl_scores_epi_fdr2,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_epi_fdr2 = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_epi_fdr2_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_epi_fdr2.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_epi_fdr2.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_epi_fdr2,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_epi_fdr2,'signed_bin_mat_file')
    
    
    
    
    
    ##### diff group 1 - group 2 (WWW - What)
    
    
    prepare_diff_cormat = pe.Node(Function(input_names=['group_cormat_matrix_file1','group_cormat_matrix_file2'],output_names = ['diff_group_cormat_matrix_file'],function = return_diff_group_mat),name='prepare_diff_cormat')
    
    main_workflow.connect(prepare_cormat1, 'group_cormat_matrix_file',prepare_diff_cormat,'group_cormat_matrix_file1')
    main_workflow.connect(prepare_cormat2, 'group_cormat_matrix_file',prepare_diff_cormat,'group_cormat_matrix_file2')
    
    
    ########## pairwise stats FDR (t-test) with scores_total
    pairwise_correl_scores_total_fdr_diff = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_total_fdr_diff_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_total_fdr_diff.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_total_fdr_diff.inputs.behav_score = scores_total
    
    main_workflow.connect(prepare_diff_cormat, 'diff_group_cormat_matrix_file',pairwise_correl_scores_total_fdr_diff,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_total_fdr_diff = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_total_fdr_diff_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_total_fdr_diff.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_total_fdr_diff.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_total_fdr_diff,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_total_fdr_diff,'signed_bin_mat_file')
    
    
    
    ########## pairwise stats FDR (t-test) with scores_epi
    pairwise_correl_scores_epi_fdr_diff = pe.Node(Function(input_names=['group_cormat_matrix_file','behav_score','correl_thresh_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_correl_stats_fdr),name='pairwise_correl_scores_epi_fdr_diff_'+ str(t_test_thresh_fdr).replace('.','_'))
    pairwise_correl_scores_epi_fdr_diff.inputs.correl_thresh_fdr = t_test_thresh_fdr
    pairwise_correl_scores_epi_fdr_diff.inputs.behav_score = scores_epi
    
    main_workflow.connect(prepare_diff_cormat, 'diff_group_cormat_matrix_file',pairwise_correl_scores_epi_fdr_diff,'group_cormat_matrix_file')
    
    plot_pairwise_correl_scores_epi_fdr_diff = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_correl_scores_epi_fdr_diff_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_correl_scores_epi_fdr_diff.inputs.coords_file = coord_rois_file
    plot_pairwise_correl_scores_epi_fdr_diff.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_correl_scores_epi_fdr_diff,'signif_signed_adj_fdr_mat_file',plot_pairwise_correl_scores_epi_fdr_diff,'signed_bin_mat_file')
    
    
    
    
    #### Run workflow
    main_workflow.write_graph('G_cor_mat_pairwise_NBS_stats_by_group_graph.dot',graph2use='flat', format = 'svg')    
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    ########################################################################### louvain ######################################################################
    
def test_generate_igraph_colors():

    from plot_igraph import generate_igraph_colors,nb_igraph_colors
    
    generate_igraph_colors(nb_igraph_colors)
    
    
if __name__ =='__main__':
    
        run_rada_coclass_pairwise_NBS()
        