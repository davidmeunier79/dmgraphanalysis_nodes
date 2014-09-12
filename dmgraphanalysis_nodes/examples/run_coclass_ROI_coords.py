# -*- coding: utf-8 -*-
"""
7th step: Compute similarity between pairs of partitions obtained after modular partitions
"""

import sys, os
sys.path.append('../irm_analysis')

from  define_variables import *

from dmgraphanalysis_nodes.nodes.coclass import PrepareCoclass,PlotCoclass,PlotIGraphCoclass

#from dmgraphanalysis.modularity import prep_radatools,community_radatools,export_lol_mask_file
#from dmgraphanalysis.modularity import plot_igraph_modules_coclass_rada,plot_igraph_modules_coclass_rada_forced_colors

    
######################################### Datasources

##### By events conditions #####################################################
    
def create_datasource_rada_by_cond_signif_conf():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'], outfields=['mod_files','coords_files','node_corres_files']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    
    mod_files=[[graph_analysis_name,'cond',"community_rada","net_List_signif_conf.lol"]],
    coords_files= [[cor_mat_analysis_name,'cond',"merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files = [[graph_analysis_name,'cond',"prep_rada","net_List_signif_conf.net"]]
    
    ) 
    
    datasource.inputs.sort_filelist = True
    
    return datasource
    
def create_datasource_rada_by_cond():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'], outfields=['mod_files','coords_files','node_corres_files']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    
    mod_files=[[graph_analysis_name,'cond',"community_rada","Z_List.lol"]],
    coords_files= [[cor_mat_analysis_name,'cond',"merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files = [[graph_analysis_name,'cond',"prep_rada","Z_List.net"]]
    
    
    ) 
    
    datasource.inputs.sort_filelist = True
    
    return datasource
    
def create_datasource_rada_event_by_cond():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['event'], outfields=['mod_files1','coords_files1','node_corres_files1','mod_files2','coords_files2','node_corres_files2']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    
    mod_files1=[[graph_analysis_name,'event',"Hit-WWW","community_rada","Z_List.lol"]],
    coords_files1= [[cor_mat_analysis_name,'event',"Hit-WWW","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files1 = [[graph_analysis_name,'event',"Hit-WWW","prep_rada","Z_List.net"]],
    
    
    mod_files2=[[graph_analysis_name,'event',"Hit-What","community_rada","Z_List.lol"]],
    coords_files2= [[cor_mat_analysis_name,'event',"Hit-What","merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files2 = [[graph_analysis_name,'event',"Hit-What","prep_rada","Z_List.net"]]
    
    
    
    ) 
    
    datasource.inputs.sort_filelist = True
    
    return datasource
    
    
def create_datasource_rada_cond_by_event():

    datasource = pe.Node(interface=nio.DataGrabber(infields = ['cond'], outfields=['mod_files1','coords_files1','node_corres_files1','mod_files2','coords_files2','node_corres_files2']),name = 'datasource')
    
    #datasource.inputs.base_directory = change_name_nifti_path
    datasource.inputs.base_directory = nipype_analyses_path
    
    datasource.inputs.template = '%s/_cond_%s_%s_subject_num_*/%s/%s'

    datasource.inputs.template_args = dict(
    
    mod_files1=[[graph_analysis_name,"Odor",'cond',"community_rada","Z_List.lol"]],
    coords_files1= [[cor_mat_analysis_name,"Odor",'cond',"merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files1 = [[graph_analysis_name,"Odor",'cond',"prep_rada","Z_List.net"]],
    
    
    mod_files2=[[graph_analysis_name,"Recall",'cond',"community_rada","Z_List.lol"]],
    coords_files2= [[cor_mat_analysis_name,"Recall",'cond',"merge_runs","coord_rois_all_runs.txt"]],
    node_corres_files2 = [[graph_analysis_name,"Recall",'cond',"prep_rada","Z_List.net"]]
    
    
    
    ) 
    
    datasource.inputs.sort_filelist = True
    
    return datasource
    
######################################### Full analysis
    
def run_rada_coclass_by_cond():
    
    #pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    main_workflow = Workflow(name= coclass_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### infosource 
    
    infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    
    infosource.iterables = [('cond', ['Odor_Hit-WWW'])]
    #infosource.iterables = [('cond', ['Rec_Hit-WWW'])]
    #infosource.iterables = [('cond', epi_cond)]
    
    #### Data source
    #datasource = create_datasource_rada_by_cond_memory_signif_conf()
    #datasource = create_datasource_rada_by_cond_signif_conf()
    datasource = create_datasource_rada_by_cond()
    
    main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ###################################### compute sum coclass and group based coclass matrices  #####################################################
    
    #### prepare_nbs_stats_rada
    
    prepare_coclass = pe.Node(interface = PrepareCoclass(),name='prepare_coclass')
    
    #Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass')
    prepare_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
     
    main_workflow.connect(datasource, 'mod_files',prepare_coclass,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files',prepare_coclass,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files',prepare_coclass,'coords_files')
    
    ######################################################## coclassification matrices ############################################################
    
    ######### norm coclass
    plot_norm_coclass= pe.Node(interface = PlotCoclass(),name='plot_norm_coclass')
    
    plot_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    plot_norm_coclass.inputs.list_value_range = [0,100]
    
    main_workflow.connect(prepare_coclass, 'norm_coclass_matrix_file',plot_norm_coclass,'coclass_matrix_file')
    
    
    
    
    
    
    
    plot_igraph_norm_coclass= pe.Node(interface = PlotIGraphCoclass(),name='plot_igraph_norm_coclass')
    
    #Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file'],output_names = ['plot_igraph_3D_sum_norm_coclass_matrix_file',],function = plot_igraph_coclass_matrix_labels
    plot_igraph_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    plot_igraph_norm_coclass.inputs.threshold = 50
    #plot_igraph_norm_coclass.inputs.threshold = 100
    plot_igraph_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    main_workflow.connect(prepare_coclass, 'norm_coclass_matrix_file',plot_igraph_norm_coclass,'coclass_matrix_file')
    
    ##### reorder_norm coclass
    #reorder_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','info_file','method_hie'],output_names = ['reordered_coclass_matrix_file','node_order_vect_file','reordered_labels_file','reordered_info_file'],function = reorder_hclust_matrix_labels),name='reorder_norm_coclass')
    #reorder_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #reorder_norm_coclass.inputs.method_hie = method_hie
    #reorder_norm_coclass.inputs.info_file = info_rois_file
    
    #main_workflow.connect(prepare_coclass,'norm_coclass_matrix_file',reorder_norm_coclass,'coclass_matrix_file')
    
    ##### plot reordered coclass
    #plot_reordered_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file'],output_names = ['plot_reordered_norm_coclass_matrix_file'],function = plot_coclass_matrix_labels),name='plot_reordered_norm_coclass')
    
    #main_workflow.connect(reorder_norm_coclass, 'reordered_coclass_matrix_file',plot_reordered_norm_coclass,'coclass_matrix_file')
    #main_workflow.connect(reorder_norm_coclass, 'reordered_labels_file',plot_reordered_norm_coclass,'labels_file')
    
    
    
    ############################################################### modular decomposition on norm_coclass ######################################################################################
    
    ########### serie 
    
    #### compute Z_list from coclass matrix
    #compute_list_norm_coclass = pe.Node(Function(input_names=['coclass_matrix_file','threshold'],output_names = ['net_list_file'],function = norm_coclass_to_net_list_thr),name='compute_list_norm_coclass')
    #compute_list_norm_coclass.inputs.threshold = 50
    
    #main_workflow.connect(prepare_coclass,'norm_coclass_matrix_file',compute_list_norm_coclass,'coclass_matrix_file')

    #### prepare Z_list for radatools processing  
    #prep_rada_norm_coclass = pe.Node(Function(input_names=['List_net_file','radatools_prep_path'],output_names = ['Pajek_net_file'],function = prep_radatools),name='prep_rada_norm_coclass')
    #prep_rada_norm_coclass.inputs.radatools_prep_path = radatools_prep_path
    
    #main_workflow.connect(compute_list_norm_coclass, 'net_list_file', prep_rada_norm_coclass, 'List_net_file')
    
    #### compute community with radatools
    #community_rada_norm_coclass = pe.Node(Function(input_names=['Pajek_net_file','optim_seq','radatools_comm_path'],output_names = ['rada_lol_file','rada_log_file'],function = community_radatools),name='community_rada_norm_coclass')
    #community_rada_norm_coclass.inputs.optim_seq = "WN trfr 100"
    ##community_rada_norm_coclass.inputs.optim_seq = "WN rfr 1"
    #community_rada_norm_coclass.inputs.radatools_comm_path = radatools_comm_path
    
    #main_workflow.connect( prep_rada_norm_coclass, 'Pajek_net_file',community_rada_norm_coclass,'Pajek_net_file')
    
    #plot_igraph_modules_rada_norm_coclass = pe.Node(Function(input_names=['rada_lol_file','Pajek_net_file','gm_mask_coords_file','labels_file'],output_names = ['coclass_all_modules_file'],function = plot_igraph_modules_coclass_rada),name='plot_igraph_modules_rada_norm_coclass')
    #plot_igraph_modules_rada_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_modules_rada_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    #main_workflow.connect(prep_rada_norm_coclass, 'Pajek_net_file',plot_igraph_modules_rada_norm_coclass,'Pajek_net_file')
    #main_workflow.connect(community_rada_norm_coclass, 'rada_lol_file',plot_igraph_modules_rada_norm_coclass,'rada_lol_file')
    
    
    
    #### Run workflow
    main_workflow.write_graph('G_coclass_rada_by_cond_group_graph.dot',graph2use='flat', format = 'svg')    
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
#def run_coclass_diff_cond():

    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    #infosource = pe.Node(interface=IdentityInterface(fields=['event']),name="infosource")
    
    ##infosource.iterables = [('event', ['Odor'])]
    #infosource.iterables = [('event', ['Odor','Rec','Recall'])]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_event_by_cond()
    
    #main_workflow.connect(infosource,'event',datasource,'event')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #prepare_coclass1 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    #prepare_coclass1.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    
    #prepare_coclass2 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    #prepare_coclass2.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
    ######### norm coclass
    
    #### substract matrix
    #diff_norm_coclass = pe.Node(Function(input_names=['mat_file1','mat_file2'],output_names = ['diff_mat_file'],function = diff_matrix),name='diff_coclass')
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file2')
    
    #### plot diff matrix
    #plot_diff_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    #plot_diff_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_diff_norm_coclass.inputs.list_value_range = [-50,50]
    
    #main_workflow.connect(diff_norm_coclass, 'diff_mat_file',plot_diff_norm_coclass,'coclass_matrix_file')
    
    ########## plot conj pos neg
    
    #### plot graph with colored edges
    #plot_igraph_conj_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file1','coclass_matrix_file2','gm_mask_coords_file','threshold','labels_file'],output_names = ['plot_igraph_conj_coclass_matrix_file'],function = plot_igraph_conj_coclass_matrix),name='plot_igraph_conj_norm_coclass')
    
    #plot_igraph_conj_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_conj_norm_coclass.inputs.threshold = 50
    #plot_igraph_conj_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file2')
    
    
    ############# pos neg diff_matrix
    
    ##### differentiate pos and neg matrices
    ##pos_neg_diff_norm_coclass = pe.Node(Function(input_names=['diff_mat_file','threshold'],output_names = ['bin_pos_mat_file','bin_neg_mat_file'],function = pos_neg_thr_matrix),name='pos_neg_diff_norm_coclass')
    ##pos_neg_diff_norm_coclass.inputs.threshold = 25
    
    ##main_workflow.connect(diff_norm_coclass, 'diff_mat_file',pos_neg_diff_norm_coclass,'diff_mat_file')
    
    ##### plot graph with colored edges
    ##plot_igraph_filtered_pos_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_pos_norm_coclass')
    
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ###plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.threshold = 50
    ##plot_igraph_filtered_pos_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    ##plot_igraph_filtered_pos_norm_coclass.inputs.edge_color = 'Blue'
    
    ##main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_filtered_pos_norm_coclass,'coclass_matrix_file')
    ##main_workflow.connect(pos_neg_diff_norm_coclass,'bin_pos_mat_file',plot_igraph_filtered_pos_norm_coclass,'bin_mat_file')
    
    
    ##plot_igraph_filtered_neg_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_neg_norm_coclass')
    
    
    ##plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ###plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_neg_norm_coclass.inputs.threshold = 50
    ##plot_igraph_filtered_neg_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    ##plot_igraph_filtered_neg_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    ##plot_igraph_filtered_neg_norm_coclass.inputs.edge_color = 'Red'
    
    ##main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_filtered_neg_norm_coclass,'coclass_matrix_file')
    ##main_workflow.connect(pos_neg_diff_norm_coclass,'bin_neg_mat_file',plot_igraph_filtered_neg_norm_coclass,'bin_mat_file')
    
    
    ##### Run workflow 
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    
#def run_coclass_diff_event():

    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    #infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    
    ##infosource.iterables = [('event', ['Odor'])]
    #infosource.iterables = [('cond', ['Hit-WWW','Hit-What'])]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_cond_by_event()
    
    #main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #prepare_coclass1 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    #prepare_coclass1.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    
    #prepare_coclass2 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    #prepare_coclass2.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
    ######### norm coclass
    
    #### substract matrix
    #diff_norm_coclass = pe.Node(Function(input_names=['mat_file1','mat_file2'],output_names = ['diff_mat_file'],function = diff_matrix),name='diff_coclass')
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file2')
    
    #### plot diff matrix
    #plot_diff_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    #plot_diff_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_diff_norm_coclass.inputs.list_value_range = [-50,50]
    
    #main_workflow.connect(diff_norm_coclass, 'diff_mat_file',plot_diff_norm_coclass,'coclass_matrix_file')
    
    ########## plot conj pos neg
    
    #### plot graph with colored edges
    #plot_igraph_conj_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file1','coclass_matrix_file2','gm_mask_coords_file','threshold','labels_file'],output_names = ['plot_igraph_conj_coclass_matrix_file'],function = plot_igraph_conj_coclass_matrix),name='plot_igraph_conj_norm_coclass')
    
    #plot_igraph_conj_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_conj_norm_coclass.inputs.threshold = 50
    #plot_igraph_conj_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file2')
    
    
    ############# pos neg diff_matrix
    
    ##### differentiate pos and neg matrices
    ##pos_neg_diff_norm_coclass = pe.Node(Function(input_names=['diff_mat_file','threshold'],output_names = ['bin_pos_mat_file','bin_neg_mat_file'],function = pos_neg_thr_matrix),name='pos_neg_diff_norm_coclass')
    ##pos_neg_diff_norm_coclass.inputs.threshold = 25
    
    ##main_workflow.connect(diff_norm_coclass, 'diff_mat_file',pos_neg_diff_norm_coclass,'diff_mat_file')
    
    ##### plot graph with colored edges
    ##plot_igraph_filtered_pos_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_pos_norm_coclass')
    
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ###plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.threshold = 50
    ##plot_igraph_filtered_pos_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    ##plot_igraph_filtered_pos_norm_coclass.inputs.edge_color = 'Blue'
    
    ##main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_filtered_pos_norm_coclass,'coclass_matrix_file')
    ##main_workflow.connect(pos_neg_diff_norm_coclass,'bin_pos_mat_file',plot_igraph_filtered_pos_norm_coclass,'bin_mat_file')
    
    
    ##plot_igraph_filtered_neg_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_neg_norm_coclass')
    
    
    ##plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ###plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_neg_norm_coclass.inputs.threshold = 50
    ##plot_igraph_filtered_neg_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    ##plot_igraph_filtered_neg_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    ##plot_igraph_filtered_neg_norm_coclass.inputs.edge_color = 'Red'
    
    ##main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_filtered_neg_norm_coclass,'coclass_matrix_file')
    ##main_workflow.connect(pos_neg_diff_norm_coclass,'bin_neg_mat_file',plot_igraph_filtered_neg_norm_coclass,'bin_mat_file')
    
    
    ##### Run workflow 
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    
    
#def run_filter_coclass_diff_cond():
    
    
    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    #infosource = pe.Node(interface=IdentityInterface(fields=['event']),name="infosource")
    
    ##infosource.iterables = [('cond', ['Odor_Hit-WWW'])]
    ##infosource.iterables = [('cond', ['Rec_Hit-WWW'])]
    #infosource.iterables = [('event', ['Odor','Rec','Recall'])]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_event_by_cond()
    
    #main_workflow.connect(infosource,'event',datasource,'event')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #prepare_coclass1 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    #prepare_coclass1.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    
    #prepare_coclass2 = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    #prepare_coclass2.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
    ########## norm coclass
    
    #### substract matrix
    #diff_norm_coclass = pe.Node(Function(input_names=['mat_file1','mat_file2'],output_names = ['diff_mat_file'],function = diff_matrix),name='diff_coclass')
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file2')
    
    #### plot diff matrix
    #plot_filtered_diff_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','filtered_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_filtered_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    #plot_filtered_diff_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_filtered_diff_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    #plot_filtered_diff_norm_coclass.inputs.list_value_range = [-50,50]
    
    #main_workflow.connect(diff_norm_coclass, 'diff_mat_file',plot_filtered_diff_norm_coclass,'coclass_matrix_file')
    
    #### pos neg diff_matrix
    
    #### differentiate pos and neg matrices
    #pos_neg_diff_norm_coclass = pe.Node(Function(input_names=['diff_mat_file','threshold'],output_names = ['bin_pos_mat_file','bin_neg_mat_file'],function = pos_neg_thr_matrix),name='pos_neg_diff_norm_coclass')
    #pos_neg_diff_norm_coclass.inputs.threshold = 25
    
    #main_workflow.connect(diff_norm_coclass, 'diff_mat_file',pos_neg_diff_norm_coclass,'diff_mat_file')
    
    
    
    
    #### plot graph with colored edges
    #plot_igraph_filtered_pos_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_pos_norm_coclass')
    
    #plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_filtered_pos_norm_coclass.inputs.threshold = 50
    #plot_igraph_filtered_pos_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_igraph_filtered_pos_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    #plot_igraph_filtered_pos_norm_coclass.inputs.edge_color = 'Blue'
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_filtered_pos_norm_coclass,'coclass_matrix_file')
    #main_workflow.connect(pos_neg_diff_norm_coclass,'bin_pos_mat_file',plot_igraph_filtered_pos_norm_coclass,'bin_mat_file')
    
    
    #plot_igraph_filtered_neg_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file','bin_mat_file','edge_color'],output_names = ['plot_igraph_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels_bin_mat_color),name='plot_igraph_filtered_neg_norm_coclass')
    
    
    #plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
    ##plot_igraph_filtered_neg_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_filtered_neg_norm_coclass.inputs.threshold = 50
    #plot_igraph_filtered_neg_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_igraph_filtered_neg_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    #plot_igraph_filtered_neg_norm_coclass.inputs.edge_color = 'Red'
    
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_filtered_neg_norm_coclass,'coclass_matrix_file')
    #main_workflow.connect(pos_neg_diff_norm_coclass,'bin_neg_mat_file',plot_igraph_filtered_neg_norm_coclass,'bin_mat_file')
    
    
    ##### Run workflow 
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    

    
    
#def run_filter_coclass_by_cond():
    
    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    #infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    
    ##infosource.iterables = [('cond', ['Odor_Hit-WWW'])]
    ##infosource.iterables = [('cond', ['Rec_Hit-WWW'])]
    #infosource.iterables = [('cond', epi_cond)]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_by_cond()
    
    #main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #prepare_coclass = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass')
    #prepare_coclass.inputs.gm_mask_coords_file = ROI_coords_np_coords_file
     
    #main_workflow.connect(datasource, 'mod_files',prepare_coclass,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files',prepare_coclass,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files',prepare_coclass,'coords_files')
    
    ######################################################### coclassification matrices ############################################################
        
    ########################################### filtered by nodes
    
    ########## norm coclass
    #plot_filtered_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','filtered_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_filtered_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    #plot_filtered_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_filtered_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    #plot_filtered_norm_coclass.inputs.list_value_range = [0,75]
    
    #main_workflow.connect(prepare_coclass, 'norm_coclass_matrix_file',plot_filtered_norm_coclass,'coclass_matrix_file')
    
    #plot_igraph_filtered_norm_coclass= pe.Node(Function(input_names=['coclass_matrix_file','gm_mask_coords_file','threshold','labels_file','filtered_file'],output_names = ['plot_igraph_3D_sum_filtered_norm_coclass_matrix_file'],function = plot_igraph_filtered_coclass_matrix_labels),name='plot_igraph_filtered_norm_coclass')
    #plot_igraph_filtered_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_filtered_norm_coclass.inputs.threshold = 50
    ##plot_igraph_filtered_norm_coclass.inputs.threshold = 100
    #plot_igraph_filtered_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_igraph_filtered_norm_coclass.inputs.filtered_file = ROI_coords_filter_file
    
    #main_workflow.connect(prepare_coclass, 'norm_coclass_matrix_file',plot_igraph_filtered_norm_coclass,'coclass_matrix_file')
    
    
    ##### Run workflow
    ##main_workflow.write_graph('G_filter_coclass_by_cond_graph.dot',graph2use='flat', format = 'svg')    
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    
#def run_reorder_rada_coclass_by_cond():
    
    ##pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    
    #infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    #infosource.iterables = [('cond', epi_cond)]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_by_cond()
    
    #main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #prepare_coclass = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass')
    #prepare_coclass.inputs.gm_mask_coords_file = coord_rois_file
     
    #main_workflow.connect(datasource, 'mod_files',prepare_coclass,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files',prepare_coclass,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files',prepare_coclass,'coords_files')
    
    ######################################################### coclassification matrices ############################################################
    
    ########### sum coclass 
    
    ##plot_coclass_with_order= pe.Node(Function(input_names=['coclass_matrix_file','node_order_vect_file','labels_file'],output_names = ['plot_coclass_matrix_file'],function = plot_order_coclass_matrix_labels),name='plot_coclass_with_order')
    ##plot_coclass_with_order.inputs.coords_file = coords_rois_file
    ##plot_coclass_with_order.inputs.node_order_vect_file = force_order_file
    
    ##main_workflow.connect(prepare_coclass,'sum_coclass_matrix_file',plot_coclass_with_order,'coclass_matrix_file')
    
    #reorder_with_force_order = pe.Node(Function(input_names=['coclass_matrix_file','labels_file','coords_file','node_order_vect_file'],output_names = ['reordered_coclass_matrix_file','reordered_labels_file','reordered_coords_file'],function = reorder_coclass_matrix_labels_coords_with_force_order),name='reorder_with_force_order')
    #reorder_with_force_order.inputs.labels_file = label_jane_rois_file
    #reorder_with_force_order.inputs.coords_file = MNI_coord_rois_file
    #reorder_with_force_order.inputs.node_order_vect_file = force_order_file
    
    #main_workflow.connect(prepare_coclass,'sum_coclass_matrix_file',reorder_with_force_order,'coclass_matrix_file')
    
    
    #plot_coclass_with_order= pe.Node(Function(input_names=['coclass_matrix_file','labels_file'],output_names = ['plot_coclass_matrix_file'],function = plot_coclass_matrix_labels),name='plot_coclass_with_order')
    
    #main_workflow.connect(reorder_with_force_order,'reordered_coclass_matrix_file',plot_coclass_with_order,'coclass_matrix_file')
    #main_workflow.connect(reorder_with_force_order,'reordered_labels_file',plot_coclass_with_order,'labels_file')
    
    
    ########### norm coclass 
    #reorder_norm_with_force_order = pe.Node(Function(input_names=['coclass_matrix_file','labels_file','coords_file','node_order_vect_file'],output_names = ['reordered_coclass_matrix_file','reordered_labels_file','reordered_coords_file'],function = reorder_coclass_matrix_labels_coords_with_force_order),name='reorder_norm_with_force_order')
    #reorder_norm_with_force_order.inputs.labels_file = label_jane_rois_file
    #reorder_norm_with_force_order.inputs.coords_file = MNI_coord_rois_file
    #reorder_norm_with_force_order.inputs.node_order_vect_file = force_order_file
    
    #main_workflow.connect(prepare_coclass,'norm_coclass_matrix_file',reorder_norm_with_force_order,'coclass_matrix_file')
    
    #plot_norm_coclass_with_order= pe.Node(Function(input_names=['coclass_matrix_file','labels_file','list_value_range'],output_names = ['plot_norm_coclass_matrix_file'],function = plot_coclass_matrix_labels_range),name='plot_norm_coclass_with_order')
    
    #plot_norm_coclass_with_order.inputs.list_value_range = [0,100]
    #main_workflow.connect(reorder_norm_with_force_order,'reordered_coclass_matrix_file',plot_norm_coclass_with_order,'coclass_matrix_file')
    #main_workflow.connect(reorder_norm_with_force_order,'reordered_labels_file',plot_norm_coclass_with_order,'labels_file')
    
    
    
    
    
    
    
    
    ##### Run workflow 
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
#def run_forcecoclassmod_coclass_by_cond():

    
    ##pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    #main_workflow = Workflow(name= coclass_analysis_name)
    #main_workflow.base_dir = nipype_analyses_path
    
    ##### infosource 
    
    #infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    #infosource.iterables = [('cond', ['Odor_Hit-WWW'])]
    ##infosource.iterables = [('cond', epi_cond)]
    
    ##### Data source
    ##datasource = create_datasource_rada_by_cond_signif_conf()
    #datasource = create_datasource_rada_by_cond()
    
    #main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ###################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_nbs_stats_rada
    
    #coclass_by_coclassmod = pe.MapNode(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file','lol_coclassmod_file','node_corres_coclassmod_file'],output_names = ['density_coclass_by_mod_by_subj_file'], function = compute_coclass_by_coclassmod),iterfield = ['mod_files','coords_files','node_corres_files'], name='coclass_by_coclassmod')
    #coclass_by_coclassmod.inputs.gm_mask_coords_file = coord_rois_file
    
    #coclass_by_coclassmod.inputs.lol_coclassmod_file = lol_coclassmod_file
    #coclass_by_coclassmod.inputs.node_corres_coclassmod_file = node_corres_coclassmod_file
    
    
    #main_workflow.connect(datasource, 'mod_files',coclass_by_coclassmod,'mod_files')
    #main_workflow.connect(datasource, 'node_corres_files',coclass_by_coclassmod,'node_corres_files')
    #main_workflow.connect(datasource, 'coords_files',coclass_by_coclassmod,'coords_files')
    
    
    
    ###### prepare_nbs_stats_rada
    
    ##coclass_by_coclassmod = pe.Node(Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file','lol_coclassmod_file','node_corres_coclassmod_file'],output_names = ['density_coclass_by_mod_by_subj_file'],function = compute_coclass_by_coclassmod),name='coclass_by_coclassmod')
    ##coclass_by_coclassmod.inputs.gm_mask_coords_file = coord_rois_file
    
    ##coclass_by_coclassmod.inputs.lol_coclassmod_file = lol_coclassmod_file
    ##coclass_by_coclassmod.inputs.node_corres_coclassmod_file = node_corres_coclassmod_file
    
    
    ##main_workflow.connect(datasource, 'mod_files',coclass_by_coclassmod,'mod_files')
    ##main_workflow.connect(datasource, 'node_corres_files',coclass_by_coclassmod,'node_corres_files')
    ##main_workflow.connect(datasource, 'coords_files',coclass_by_coclassmod,'coords_files')
    
    
    
    
    ##### Run workflow 
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
if __name__ =='__main__':
    
    #test_generate_igraph_colors()
    
    ############# run all 
    
    print split_coclass_analysis_name
    
    if 'forceorder' in split_coclass_analysis_name:
        run_reorder_rada_coclass_by_cond()
    elif 'forcecoclassmod' in split_coclass_analysis_name:
        
        print "run forcecoclassmod"
        run_forcecoclassmod_coclass_by_cond()
        
    elif 'filter' in split_coclass_analysis_name:
        
        if 'diff' in split_coclass_analysis_name and 'cond' in split_coclass_analysis_name:
                    
            print "run filter diff_cond"
            run_filter_coclass_diff_cond()
            
        else:
            
            print "run filter"
            run_filter_coclass_by_cond()
        
        
        
    else:
        if 'diff' in split_coclass_analysis_name:
        
            if 'cond' in split_coclass_analysis_name:
                        
                print "run diff_cond"
                run_coclass_diff_cond()
                
            elif 'event' in split_coclass_analysis_name:
                        
                print "run diff_event"
                run_coclass_diff_event()
            
        else:
            print "run rada"
            run_rada_coclass_by_cond()
        
        