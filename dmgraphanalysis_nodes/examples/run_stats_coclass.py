# -*- coding: utf-8 -*-
"""
7th step: Compute similarity between pairs of partitions obtained after modular partitions
"""

import sys, os
sys.path.append('../irm_analysis')

from define_variables import *

### cette partie pourrait aller dans utils_plot.py ....
from dmgraphanalysis.graph_stats import plot_signed_bin_mat_labels,plot_signed_bin_mat_labels_only_fdr,plot_signed_bin_mat,plot_bin_mat,plot_img_val_vect

from dmgraphanalysis.graph_stats import prepare_nbs_stats_rada, compute_pairwise_ttest_stats_fdr,compute_pairwise_binom_stats_fdr,compute_nodewise_ttest_stats_fdr

################################### datasource #######################################

##### By event conditions (Hit_WWW vs. Hit_What) #####################################################
    
def create_datasource_rada_by_cond_event_signif_conf():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'],outfields=['mod_files1','coords_files1','node_corres_files1','mod_files2','coords_files2','node_corres_files2']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    mod_files1=[[graph_analysis_name,'cond',"Hit-WWW","community_rada","net_List_signif_conf.lol"]],
    coords_files1= [[cor_mat_analysis_name,'cond',"Hit-WWW","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files1 = [[graph_analysis_name,'cond',"Hit-WWW","prep_rada","net_List_signif_conf.net"]],
    
    mod_files2=[[graph_analysis_name,'cond',"Hit-What","community_rada","net_List_signif_conf.lol"]],
    coords_files2= [[cor_mat_analysis_name,'cond',"Hit-What","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files2 = [[graph_analysis_name,'cond',"Hit-What","prep_rada","net_List_signif_conf.net"]]    
    ) 
    datasource.inputs.sort_filelist = True
    
    return datasource
    
def create_datasource_rada_by_cond_event():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'],outfields=['mod_files1','coords_files1','node_corres_files1','mod_files2','coords_files2','node_corres_files2']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    mod_files1=[[graph_analysis_name,'cond',"Hit-WWW","community_rada","Z_List.lol"]],
    coords_files1= [[cor_mat_analysis_name,'cond',"Hit-WWW","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files1 = [[graph_analysis_name,'cond',"Hit-WWW","prep_rada","Z_List.net"]],
    
    mod_files2=[[graph_analysis_name,'cond',"Hit-What","community_rada","Z_List.lol"]],
    coords_files2= [[cor_mat_analysis_name,'cond',"Hit-What","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files2 = [[graph_analysis_name,'cond',"Hit-What","prep_rada","Z_List.net"]]    
    ) 
    datasource.inputs.sort_filelist = True
    
    return datasource
##### By memory conditions (memory_of_WWW-S3 vs.  memory_of_What-S3) #####################################################
    
def create_datasource_rada_by_cond_memory_signif_conf():

    datasource = pe.Node(interface=nio.DataGrabber(outfields=['mod_files1','coords_files1','node_corres_files1','mod_files2','coords_files2','node_corres_files2']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    mod_files1=[[graph_analysis_name,"memory_of_WWW-S3","community_rada","net_List_signif_conf.lol"]],
    coords_files1= [[cor_mat_analysis_name,"memory_of_WWW-S3","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files1 = [[graph_analysis_name,"memory_of_WWW-S3","prep_rada","net_List_signif_conf.net"]],
    
    mod_files2=[[graph_analysis_name,"memory_of_What-S3","community_rada","net_List_signif_conf.lol"]],
    coords_files2= [[cor_mat_analysis_name,"memory_of_What-S3","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files2 = [[graph_analysis_name,"memory_of_What-S3","prep_rada","net_List_signif_conf.net"]]    
    ) 
    datasource.inputs.sort_filelist = True
    
    return datasource
    
def run_rada_coclass_pairwise_NBS():
    
    #pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    
    main_workflow = Workflow(name= stat_coclass_analysis_name)
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
        datasource = create_datasource_rada_by_cond_event_signif_conf()
            
        main_workflow.connect(infosource, 'cond', datasource, 'cond')
        
    elif model_name == 'Model9_Only3events_WWW_What':
    
        #### Info source
        infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
        
        ### all 
        infosource.iterables = [('cond', ['Odor','Rec','Recall'])]
        #infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor','Rec','Recall'])]
        #infosource.iterables = [('cond', ['Rest','Odor_anticipation','Odor_preparation','Odor','Rec','Recall'])]
        
        #### Data source
        #datasource = create_datasource_rada_by_cond_event_signif_conf()
        datasource = create_datasource_rada_by_cond_event()
            
        main_workflow.connect(infosource, 'cond', datasource, 'cond')
    
    elif model_name == 'Model8_Memory_Hit':
        
        #### Data source
        datasource = create_datasource_rada_by_cond_memory_signif_conf()
            
        
        
    ###################################### compute sum coclass and group based coclass matrices  #####################################################
    
    #### prepare_nbs_stats_rada
    
    prepare_coclass1 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','group_nod_modular_domain_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    prepare_coclass1.inputs.gm_mask_coords_file = coord_rois_file
    #prepare_coclass1.inputs.gm_mask_file = os.path.join(main_path,dartel_analysis_name ,"gm_mask","gm_mask.nii")
    
    main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    prepare_coclass2 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','group_nod_modular_domain_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    prepare_coclass2.inputs.gm_mask_coords_file = coord_rois_file
    #prepare_coclass2.inputs.gm_mask_file = os.path.join(main_path,dartel_analysis_name ,"gm_mask","gm_mask.nii")
    
    main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
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
    
    ########## pairwise stats FDR (binomial test)
    pairwise_stats_fdr = pe.Node(Function(input_names=['group_coclass_matrix_file1','group_coclass_matrix_file2','conf_interval_binom_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_binom_stats_fdr),name='pairwise_stats_fdr')
    pairwise_stats_fdr.inputs.conf_interval_binom_fdr = conf_interval_binom_fdr
    main_workflow.connect(prepare_coclass1, 'group_coclass_matrix_file',pairwise_stats_fdr,'group_coclass_matrix_file1')
    main_workflow.connect(prepare_coclass2, 'group_coclass_matrix_file',pairwise_stats_fdr,'group_coclass_matrix_file2')
    
    plot_pairwise_stats = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels),name='plot_pairwise_stats')
    plot_pairwise_stats.inputs.coords_file = coord_rois_file
    plot_pairwise_stats.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats,'signed_bin_mat_file')
    
    plot_pairwise_stats_fdr = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels_only_fdr),name='plot_pairwise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_pairwise_stats_fdr.inputs.coords_file = coord_rois_file
    plot_pairwise_stats_fdr.inputs.labels_file = label_jane_rois_file
    
    main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats_fdr,'signed_bin_mat_file')
    
    
    #plot_pairwise_stats_fdr = pe.Node(Function(input_names=['signed_bin_mat_file','coords_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat),name='plot_pairwise_stats_fdr')
    #plot_pairwise_stats_fdr.inputs.coords_file = coord_rois_file
    
    #main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats_fdr,'signed_bin_mat_file')
    
    
    ########### pairwise stats (binomial test)
    #pairwise_stats = pe.Node(Function(input_names=['group_coclass_matrix_file1','group_coclass_matrix_file2'],output_names = ['pairwise_binom_adj_mat_file','degree_pairwise_binom_adj_mat_file'],function = compute_coclass_pairwise_binom_stats),name='pairwise_stats_'+ str(conf_interval_binom).replace('.','_'))
    
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
    #nodewise_stats = pe.Node(Function(input_names=['group_nod_modular_domain_file1','group_nod_modular_domain_file2'],output_names = ['nodewise_t_val_vect_file'],function = compute_coclass_nodewise_ttest_stats),name='nodewise_stats_'+ str(THRESH).replace('.','_'))
    
    #main_workflow.connect(prepare_coclass1, 'group_nod_modular_domain_file',nodewise_stats,'group_nod_modular_domain_file1')
    #main_workflow.connect(prepare_coclass2, 'group_nod_modular_domain_file',nodewise_stats,'group_nod_modular_domain_file2')
    
    
    #plot_nodewise_stats = pe.Node(Function(input_names=['val_vect_file','indexed_mask_file'],output_names = ['val_vect_img_file'],function = plot_img_val_vect),name='plot_nodewise_stats_'+ str(THRESH).replace('.','_'))
    #plot_nodewise_stats.inputs.indexed_mask_file = indexed_mask_rois_file
    
    #main_workflow.connect(nodewise_stats,'nodewise_t_val_vect_file',plot_nodewise_stats,'val_vect_file')
    
    
    ############ nodewise stats fdr (t-test) 
    nodewise_stats_fdr = pe.Node(Function(input_names=['group_vect_file1','group_vect_file2','t_test_thresh_fdr'],output_names = ['nodewise_t_val_vect_file'],function = compute_nodewise_ttest_stats_fdr),name='nodewise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    nodewise_stats_fdr.inputs.t_test_thresh_fdr = t_test_thresh_fdr
    
    main_workflow.connect(prepare_coclass1, 'group_nod_modular_domain_file',nodewise_stats_fdr,'group_vect_file1')
    main_workflow.connect(prepare_coclass2, 'group_nod_modular_domain_file',nodewise_stats_fdr,'group_vect_file2')
    
    
    plot_nodewise_stats_fdr = pe.Node(Function(input_names=['val_vect_file','indexed_mask_file'],output_names = ['val_vect_img_file'],function = plot_img_val_vect),name='plot_nodewise_stats_fdr_'+ str(t_test_thresh_fdr).replace('.','_'))
    plot_nodewise_stats_fdr.inputs.indexed_mask_file = indexed_mask_rois_file
    
    main_workflow.connect(nodewise_stats_fdr,'nodewise_t_val_vect_file',plot_nodewise_stats_fdr,'val_vect_file')
    
    #### Run workflow
    main_workflow.write_graph(stat_coclass_analysis_name + '_graph.dot',graph2use='flat', format = 'svg')    
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
        