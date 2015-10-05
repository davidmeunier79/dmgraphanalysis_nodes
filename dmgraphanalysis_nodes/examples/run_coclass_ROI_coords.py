# -*- coding: utf-8 -*-
"""
7th step: Compute similarity between pairs of partitions obtained after modular partitions
"""

import sys, os
sys.path.append('../irm_analysis')

from  define_variables import *

from dmgraphanalysis_nodes.nodes.coclass import PrepareCoclass,PlotCoclass,PlotIGraphCoclass,DiffMatrices,PlotIGraphConjCoclass
from dmgraphanalysis_nodes.nodes.modularity import ComputeIntNetList, PrepRada, CommRada, PlotIGraphModules, ComputeNodeRoles, NetPropRada

from dmgraphanalysis_nodes.nodes.graph_stats import StatsPairBinomial,PlotIGraphSignedIntMat

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
    
def run_coclass_by_cond():
    
    #pairwise_subj_indexes = [pair for pair in it.combinations(['P06','P07','A08'],2)]
    
    main_workflow = Workflow(name= coclass_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### infosource 
    
    infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    
    #infosource.iterables = [('cond', ['Odor_Hit-WWW'])]
    #infosource.iterables = [('cond', ['Recall_Hit-What'])]
    infosource.iterables = [('cond', epi_cond)]
    
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
    plot_igraph_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    plot_igraph_norm_coclass.inputs.threshold = 50
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
    
    if 'rada' in split_coclass_analysis_name:
        
        ### compute Z_list from coclass matrix
        compute_list_norm_coclass = pe.Node(interface = ComputeIntNetList(),name='compute_list_norm_coclass')
        compute_list_norm_coclass.inputs.threshold = 50
        
        main_workflow.connect(prepare_coclass,'norm_coclass_matrix_file',compute_list_norm_coclass, 'int_mat_file')
        
        
        #################################################### radatools ################################################################

        ### prepare net_list for radatools processing  
        prep_rada = pe.Node(interface = PrepRada(),name='prep_rada')
        prep_rada.inputs.radatools_path = radatools_path
        
        main_workflow.connect(compute_list_norm_coclass, 'net_List_file', prep_rada, 'net_List_file')
        
        ### compute community with radatools
        community_rada = pe.Node(interface = CommRada(), name='community_rada')
        community_rada.inputs.optim_seq = radatools_optim
        community_rada.inputs.radatools_path = radatools_path
        
        main_workflow.connect( prep_rada, 'Pajek_net_file',community_rada,'Pajek_net_file')
        
            

        ### node roles
        node_roles = pe.Node(interface = ComputeNodeRoles(role_type = "4roles"), name='node_roles')
        
        main_workflow.connect( prep_rada, 'Pajek_net_file',node_roles,'Pajek_net_file')
        main_workflow.connect( community_rada, 'rada_lol_file',node_roles,'rada_lol_file')
        

            
        #### plot_igraph_modules_rada_norm_coclass
        
        plot_igraph_modules_rada_norm_coclass = pe.Node(interface = PlotIGraphModules(),name='plot_igraph_modules_rada_norm_coclass')
        
        plot_igraph_modules_rada_norm_coclass.inputs.labels_file = ROI_coords_labels_file
        plot_igraph_modules_rada_norm_coclass.inputs.coords_file = ROI_coords_MNI_coords_file
        
        main_workflow.connect(prep_rada, 'Pajek_net_file',plot_igraph_modules_rada_norm_coclass,'Pajek_net_file')
        main_workflow.connect(community_rada, 'rada_lol_file',plot_igraph_modules_rada_norm_coclass,'rada_lol_file')
        
        main_workflow.connect(node_roles, 'node_roles_file',plot_igraph_modules_rada_norm_coclass,'node_roles_file')
        
        
        
        ############ compute networ properties with rada
        
        net_prop = pe.Node(interface = NetPropRada(optim_seq = "A"), name = 'net_prop')
        net_prop.inputs.radatools_path = radatools_path
        
        main_workflow.connect(prep_rada, 'Pajek_net_file',net_prop,'Pajek_net_file')
        
        
        
    #### Run workflow
    main_workflow.write_graph('G_coclass_by_cond_group_graph.dot',graph2use='flat', format = 'svg')    
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
def run_coclass_diff_cond():

    main_workflow = Workflow(name= coclass_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### infosource 
    infosource = pe.Node(interface=IdentityInterface(fields=['event']),name="infosource")
    
    #infosource.iterables = [('event', ['Odor'])]
    infosource.iterables = [('event', ['Odor','Rec','Recall'])]
    
    #### Data source
    #datasource = create_datasource_rada_by_cond_signif_conf()
    datasource = create_datasource_rada_event_by_cond()
    
    main_workflow.connect(infosource,'event',datasource,'event')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_coclass
    prepare_coclass1 = pe.Node(interface = PrepareCoclass(),name='prepare_coclass1')
    
    #Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    prepare_coclass1.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
     
    main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    
    prepare_coclass2 = pe.Node(interface = PrepareCoclass(),name='prepare_coclass2')
    
    #Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    prepare_coclass2.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
     
    main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
    ########## norm coclass
    
    #### substract matrix
    #diff_norm_coclass = pe.Node(interface = DiffMatrices(),name='diff_coclass')
    ##Function(input_names=['mat_file1','mat_file2'],output_names = ['diff_mat_file'],function = diff_matrix),name='diff_coclass')
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file2')
    
    #### plot diff matrix
    #plot_diff_norm_coclass= pe.Node(interface = PlotCoclass(),name='plot_diff_norm_coclass')
    
    ##Function(input_names=['coclass_matrix_file','labels_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    #plot_diff_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    #plot_diff_norm_coclass.inputs.list_value_range = [-50,50]
    
    #main_workflow.connect(diff_norm_coclass, 'diff_mat_file',plot_diff_norm_coclass,'coclass_matrix_file')
    
    ########## plot conj pos neg
    
    #### plot graph with colored edges
    #plot_igraph_conj_norm_coclass= pe.Node(interface = PlotIGraphConjCoclass(),name='plot_igraph_conj_norm_coclass')
    
    ##Function(input_names=['coclass_matrix_file1','coclass_matrix_file2','gm_mask_coords_file','threshold','labels_file'],output_names = ['plot_igraph_conj_coclass_matrix_file'],function = plot_igraph_conj_coclass_matrix),name='plot_igraph_conj_norm_coclass')
    
    #plot_igraph_conj_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    ##plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_conj_norm_coclass.inputs.threshold = 50
    #plot_igraph_conj_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    #main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file1')
    #main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file2')
    
    
    ####################################################################################### Stats computation on coclass matrices ############################################
    
    
    
    ########## pairwise stats FDR (binomial test)
    pairwise_stats_fdr = pe.Node(interface = StatsPairBinomial(),name='pairwise_stats_fdr')
    #Function(input_names=['group_coclass_matrix_file1','group_coclass_matrix_file2','conf_interval_binom_fdr'],output_names = ['signif_signed_adj_fdr_mat_file'],function = compute_pairwise_binom_stats_fdr),name='pairwise_stats_fdr')
    pairwise_stats_fdr.inputs.conf_interval_binom_fdr = conf_interval_binom_fdr
    
    main_workflow.connect(prepare_coclass1, 'group_coclass_matrix_file',pairwise_stats_fdr,'group_coclass_matrix_file1')
    main_workflow.connect(prepare_coclass2, 'group_coclass_matrix_file',pairwise_stats_fdr,'group_coclass_matrix_file2')
    
    plot_pairwise_stats = pe.Node(interface = PlotIGraphSignedIntMat(),name='plot_pairwise_stats')
    
    #Function(input_names=['signed_bin_mat_file','coords_file','labels_file'],output_names = ['plot_3D_bin_mat_file','heatmap_bin_mat_file'],function = plot_signed_bin_mat_labels),name='plot_pairwise_stats')
    plot_pairwise_stats.inputs.coords_file = ROI_coords_MNI_coords_file
    plot_pairwise_stats.inputs.labels_file = ROI_coords_labels_file
    
    main_workflow.connect(pairwise_stats_fdr,'signif_signed_adj_fdr_mat_file',plot_pairwise_stats,'signed_int_mat_file')
    
    
    
    
    
    
    
    
    
    
    
    #### Run workflow 
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    
def run_coclass_diff_event():

    main_workflow = Workflow(name= coclass_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### infosource 
    infosource = pe.Node(interface=IdentityInterface(fields=['cond']),name="infosource")
    
    #infosource.iterables = [('event', ['Odor'])]
    infosource.iterables = [('cond', ['Hit-WWW','Hit-What'])]
    
    #### Data source
    #datasource = create_datasource_rada_by_cond_signif_conf()
    datasource = create_datasource_rada_cond_by_event()
    
    main_workflow.connect(infosource,'cond',datasource,'cond')
    
    ####################################### compute sum coclass and group based coclass matrices  #####################################################
    
    ##### prepare_coclass
    prepare_coclass1 = pe.Node(interface = PrepareCoclass(),name='prepare_coclass1')
    
    #Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass1')
    prepare_coclass1.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
     
    main_workflow.connect(datasource, 'mod_files1',prepare_coclass1,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files1',prepare_coclass1,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files1',prepare_coclass1,'coords_files')
    
    
    prepare_coclass2 = pe.Node(interface = PrepareCoclass(),name='prepare_coclass2')
    
    #Function(input_names=['mod_files','coords_files','node_corres_files','gm_mask_coords_file'],output_names = ['group_coclass_matrix_file','sum_coclass_matrix_file','sum_possible_edge_matrix_file','norm_coclass_matrix_file'],function = prepare_nbs_stats_rada),name='prepare_coclass2')
    prepare_coclass2.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
     
    main_workflow.connect(datasource, 'mod_files2',prepare_coclass2,'mod_files')
    main_workflow.connect(datasource, 'node_corres_files2',prepare_coclass2,'node_corres_files')
    main_workflow.connect(datasource, 'coords_files2',prepare_coclass2,'coords_files')
    
    ######### norm coclass
    
    ### substract matrix
    diff_norm_coclass = pe.Node(interface = DiffMatrices(),name='diff_coclass')
    #Function(input_names=['mat_file1','mat_file2'],output_names = ['diff_mat_file'],function = diff_matrix),name='diff_coclass')
    
    main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file1')
    main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',diff_norm_coclass,'mat_file2')
    
    ### plot diff matrix
    plot_diff_norm_coclass= pe.Node(interface = PlotCoclass(),name='plot_diff_norm_coclass')
    
    #Function(input_names=['coclass_matrix_file','labels_file','list_value_range'],output_names = ['plot_hist_coclass_matrix_file','plot_coclass_matrix_file'],function = plot_coclass_matrix_labels_range),name='plot_filtered_norm_coclass')
    plot_diff_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    plot_diff_norm_coclass.inputs.list_value_range = [-50,50]
    
    main_workflow.connect(diff_norm_coclass, 'diff_mat_file',plot_diff_norm_coclass,'coclass_matrix_file')
    
    ######### plot conj pos neg
    
    ### plot graph with colored edges
    plot_igraph_conj_norm_coclass= pe.Node(interface = PlotIGraphConjCoclass(),name='plot_igraph_conj_norm_coclass')
    
    #Function(input_names=['coclass_matrix_file1','coclass_matrix_file2','gm_mask_coords_file','threshold','labels_file'],output_names = ['plot_igraph_conj_coclass_matrix_file'],function = plot_igraph_conj_coclass_matrix),name='plot_igraph_conj_norm_coclass')
    
    plot_igraph_conj_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    #plot_igraph_filtered_pos_norm_coclass.inputs.gm_mask_coords_file = ROI_coords_MNI_coords_file
    plot_igraph_conj_norm_coclass.inputs.threshold = 50
    plot_igraph_conj_norm_coclass.inputs.labels_file = ROI_coords_labels_file
    
    main_workflow.connect(prepare_coclass1,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file1')
    main_workflow.connect(prepare_coclass2,'norm_coclass_matrix_file',plot_igraph_conj_norm_coclass,'coclass_matrix_file2')
    
    #### Run workflow 
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
    
    
    
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
    
    
#def gather_coclass_results():

    #from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    #from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    #from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    #import pandas as pd
    
    #### labels
    
    #labels = [line.strip() for line in open(ROI_coords_labels_file)]


    #print 'loading ROI coords'
    
    #MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    #print MNI_coords.shape
        
        
        
        
        
    #list_coclass_mat = []
    
    #for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        #coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        #print coclass_list_file
        
        #node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        #print node_corres
        #print sparse_mat
        
        #gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        #node_coords = gm_coords[node_corres,:]
        
        #full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        #print full_cormat.shape
        
        ##full_cormat[full_cormat > 0] = 1
        
        #list_coclass_mat.append(full_cormat)
        
    #print list_coclass_mat
    
    #all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    #print all_coclass_mat
    #print all_coclass_mat.shape
    
    #signif_all_coclass_mat = np.copy(all_coclass_mat)
    #signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    
    #print signif_all_coclass_mat
    #print signif_all_coclass_mat.shape
    
    #### reseau commum aux 4 conditions
    #print "Computing core coclass mat"
    
    #core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    #core_coclass_mat[core_coclass_mat != 4] = 0
    #core_coclass_mat[core_coclass_mat == 4] = 1
    
    #print np.sum(core_coclass_mat)
    
    #### reseau commum aux 2 conditions epi
    #print "Computing epi coclass mat"
    
    #epi_coclass_mat = np.sum(signif_all_coclass_mat[(0,2),:,:],axis = 0)
    
    #print epi_coclass_mat
    
    #epi_coclass_mat[epi_coclass_mat != 2] = 0
    #epi_coclass_mat[epi_coclass_mat == 2] = 1
    
    ##epi_coclass_mat[core_coclass_mat == 1] = 0
    
    #print np.sum(epi_coclass_mat)
    
    ######### plotting core + epi
    
    #core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    #core_epi_coclass_mat[core_coclass_mat == 1] = 1
    #core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    
    #core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'core_epi_coclass.eps')
    
    #plot_3D_igraph_int_mat(core_epi_coclass_file,core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black"])
    
    #0/0
    
    
    ####reseau commun odor_WWW odor_what
    #print "Computing conj odor coclass mat"
    
    #odor_conj_coclass_mat = np.sum(signif_all_coclass_mat[(0,1),:,:],axis = 0)
    
    #odor_conj_coclass_mat[odor_conj_coclass_mat != 2] = 0
    #odor_conj_coclass_mat[odor_conj_coclass_mat == 2] = 1
    
    ##odor_conj_coclass_mat[core_coclass_mat == 1] = 0
    ##odor_conj_coclass_mat[epi_coclass_mat == 1] = 0
    
    #print np.sum(odor_conj_coclass_mat)
    
    #### reseau specif odor WWW
    
    #print "Computing specif odor WWW coclass mat"
    #odor_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    ##print odor_specif_coclass_mat
    ##print np.sum(odor_specif_coclass_mat)
    
    #odor_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    ##odor_specif_coclass_mat[odor_conj_coclass_mat == 1] = 0
    ##odor_specif_coclass_mat[core_coclass_mat == 1] = 0
    ##odor_specif_coclass_mat[epi_coclass_mat == 1] = 0
    
    
    #print np.sum(odor_specif_coclass_mat)
    
    
    
    ######### plotting core + epi + odor
    
    #odor_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    #odor_core_epi_coclass_mat[odor_specif_coclass_mat == 1] = 4
    #odor_core_epi_coclass_mat[odor_conj_coclass_mat == 1] = 3
    
    #odor_core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    #odor_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    
    #odor_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'odor_core_epi_coclass.eps')
    
    #plot_3D_igraph_int_mat(odor_core_epi_coclass_file,odor_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black","Blue","Cyan"])
    
    ####reseau commun recall_WWW recall_what
    #print "Computing conj recall coclass mat"
    
    #recall_conj_coclass_mat = np.sum(signif_all_coclass_mat[(2,3),:,:],axis = 0)
    
    #recall_conj_coclass_mat[recall_conj_coclass_mat != 2] = 0
    #recall_conj_coclass_mat[recall_conj_coclass_mat == 2] = 1
    
    ##recall_conj_coclass_mat[core_coclass_mat == 1] = 0
    ##recall_conj_coclass_mat[epi_coclass_mat == 1] = 0
    
    #print np.sum(recall_conj_coclass_mat)
    
    #### reseau specif recall WWW
    
    #print "Computing specif recall WWW coclass mat"
    #recall_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    ##print recall_specif_coclass_mat
    ##print np.sum(recall_specif_coclass_mat)
    
    #recall_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    ##recall_specif_coclass_mat[recall_conj_coclass_mat == 1] = 0
    ##recall_specif_coclass_mat[core_coclass_mat == 1] = 0
    ##recall_specif_coclass_mat[epi_coclass_mat == 1] = 0
    
    #print np.sum(recall_specif_coclass_mat)
    
    
    ######### plotting core + epi + recall
    
    #recall_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    #recall_core_epi_coclass_mat[recall_specif_coclass_mat == 1] = 4
    #recall_core_epi_coclass_mat[recall_conj_coclass_mat == 1] = 3
    
    #recall_core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    #recall_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    
    #recall_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'recall_core_epi_coclass.eps')
    
    #plot_3D_igraph_int_mat(recall_core_epi_coclass_file,recall_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black","Red","Pink"])
    
    
    #print "Computing degree"
    
    #core_degree = np.sum(core_coclass_mat,axis = 0)
    
    #print core_degree.shape
    
    #epi_degree =  np.sum(epi_coclass_mat,axis = 0)
    
    #print epi_degree.shape
    
    #odor_conj_degree =  np.sum(odor_conj_coclass_mat,axis = 0)
    
    #print odor_conj_degree.shape
    
    #odor_specif_degree =  np.sum(odor_specif_coclass_mat,axis = 0)
    
    #print odor_specif_degree.shape
    
    #recall_conj_degree =  np.sum(recall_conj_coclass_mat,axis = 0)
    
    #print recall_conj_degree.shape
    
    #recall_specif_degree =  np.sum(recall_specif_coclass_mat,axis = 0)
    
    #print recall_specif_degree.shape
    
    #print labels
        
    #tab_degree = np.column_stack((core_degree,epi_degree,odor_conj_degree,odor_specif_degree,recall_conj_degree,recall_specif_degree))
    
    #df = pd.DataFrame(tab_degree,columns = ['Core','Episodic','Conj_odor','Signif_odor','Conj_recall','signif_recall'],index = labels)
    
    #df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'degrees_node.txt')
    
    #df.to_csv(df_filename)
    
    
    #print tab_degree.shape
        
        
def gather_coclass_excluded_results():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
        
        
        
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    ### individual z-score
    #print "computing z-score"
    
    #mean_degree = np.mean(degree_all_coclass, axis = 1)
    #sd_degree = np.std(degree_all_coclass, axis = 1)
    
    #print mean_degree,sd_degree
    
    #z_score_degree_coclass = np.zeros(shape = degree_all_coclass.shape, dtype = float)
    
    #for i in range(degree_all_coclass.shape[0]):
    
        #z_score_degree_coclass[i,:] =  (degree_all_coclass[i,:] - mean_degree[i])/sd_degree[i]
    
        #print z_score_degree_coclass
        
    #print z_score_degree_coclass
    #print z_score_degree_coclass.shape
    
    #print "computing hubs"
    
    #hubs_all_coclass = z_score_degree_coclass > 1.0
    
    #print hubs_all_coclass
    
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[0,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[1,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[2,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[3,:])]
    
    ############### zscore by event (same for odor)
    #print "computing z-score"
    
    #mean_all_degree = np.mean(degree_all_coclass[(0,1),:])
    #sd_all_degree = np.std(degree_all_coclass[(0,1),:])
    
    ##print mean_all_degree,sd_all_degree
    
    #z_score_degree_all_coclass = (degree_all_coclass[(0,1),:] - mean_all_degree)/sd_all_degree
    
    ##print z_score_degree_all_coclass
    ##print z_score_degree_all_coclass.shape
    
    #print "computing hubs"
    
    #hubs_all_coclass = z_score_degree_all_coclass > 1.5
    
    ##print hubs_all_coclass
    
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[0,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[1,:])]
    
    
    ############### zscore by event (same for recall)
    #print "computing z-score"
    
    #mean_all_degree = np.mean(degree_all_coclass[(2,3),:])
    #sd_all_degree = np.std(degree_all_coclass[(2,3),:])
    
    ##print mean_all_degree,sd_all_degree
    
    #z_score_degree_all_coclass = (degree_all_coclass[(2,3),:] - mean_all_degree)/sd_all_degree
    
    ##print z_score_degree_all_coclass
    ##print z_score_degree_all_coclass.shape
    
    #print "computing hubs"
    
    #hubs_all_coclass = z_score_degree_all_coclass > 1.5
   
    ##print hubs_all_coclass
    
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[0,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[1,:])]
    
    
    
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[2,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[3,:])]
    
    
    ########## zscore same mean/std for all
    #print "computing z-score"
    
    #mean_all_degree = np.mean(degree_all_coclass)
    #sd_all_degree = np.std(degree_all_coclass)
    
    #print mean_all_degree,sd_all_degree
    
    #z_score_degree_all_coclass = (degree_all_coclass - mean_all_degree)/sd_all_degree
    
    #print z_score_degree_all_coclass
    #print z_score_degree_all_coclass.shape
    
    #print "cimputing hubs"
    
    #hubs_all_coclass = z_score_degree_all_coclass > 1.0
    
    #print hubs_all_coclass
    
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[0,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[1,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[2,:])]
    #print np.array(labels,dtype = "string")[np.where(hubs_all_coclass[3,:])]
    
    
    ### reseau commum aux 4 conditions
    print "Computing core coclass mat"
    
    core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    core_coclass_mat[core_coclass_mat != 4] = 0
    core_coclass_mat[core_coclass_mat == 4] = 1
    
    print np.sum(core_coclass_mat)
    
    core_degree = np.sum(core_coclass_mat,axis = 0)
    
    print core_degree
    print core_degree.shape
    
    ### reseau commum aux 2 conditions epi
    print "Computing epi coclass mat"
    
    epi_coclass_mat = np.sum(signif_all_coclass_mat[(0,2),:,:],axis = 0)
    
    #print epi_coclass_mat
    
    epi_coclass_mat[epi_coclass_mat != 2] = 0
    epi_coclass_mat[epi_coclass_mat == 2] = 1
    
    epi_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(epi_coclass_mat)
    
    epi_degree =  np.sum(epi_coclass_mat,axis = 0)
    
    print epi_degree.shape
    
    
    
    
    node_core_epi_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_epi_labels[4 <= core_degree] = 1
    node_core_epi_labels[4 <= epi_degree] = 2
    
    print node_core_epi_labels
    
    node_core_epi_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    node_core_epi_sizes[4 <= core_degree] = core_degree[4 <= core_degree] * 2
    node_core_epi_sizes[4 <= epi_degree] = epi_degree[4 <= epi_degree] * 2
    
    ######## plotting core + epi
    
    core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    core_epi_coclass_mat[core_coclass_mat == 1] = 1
    
    core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_epi_coclass.eps')
    
    plot_3D_igraph_int_mat(core_epi_coclass_file,core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black"],node_col_labels = node_core_epi_labels,nodes_sizes = node_core_epi_sizes)
    
    ###reseau commun odor_WWW odor_what
    print "Computing conj odor coclass mat"
    
    odor_conj_coclass_mat = np.sum(signif_all_coclass_mat[(0,1),:,:],axis = 0)
    
    odor_conj_coclass_mat[odor_conj_coclass_mat != 2] = 0
    odor_conj_coclass_mat[odor_conj_coclass_mat == 2] = 1
    
    odor_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(odor_conj_coclass_mat)
    
    odor_conj_degree =  np.sum(odor_conj_coclass_mat,axis = 0)
    
    print odor_conj_degree.shape
    
    ### reseau specif odor WWW
    
    print "Computing specif odor WWW coclass mat"
    odor_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    #print odor_specif_coclass_mat
    #print np.sum(odor_specif_coclass_mat)
    
    odor_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    odor_specif_coclass_mat[odor_conj_coclass_mat == 1] = 0
    #odor_specif_coclass_mat[core_coclass_mat == 1] = 0
    
    
    print np.sum(odor_specif_coclass_mat)
    
    
    odor_specif_degree =  np.sum(odor_specif_coclass_mat,axis = 0)
    
    print odor_specif_degree.shape
    
    
    
    
    
    node_core_odor_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_odor_labels[4 <= core_degree] = 1
    node_core_odor_labels[4 <= odor_conj_degree] = 2
    node_core_odor_labels[4 <= odor_specif_degree] = 3
    
    print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[4 <= core_degree] = core_degree[4 <= core_degree] * 2
    node_core_odor_sizes[4 <= odor_conj_degree] = odor_conj_degree[4 <= odor_conj_degree] * 2
    node_core_odor_sizes[4 <= odor_specif_degree] = odor_specif_degree[4 <= odor_specif_degree] * 2
    
    
    
    
    
    
    
    
    ######## plotting core + odor
    
    odor_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    odor_core_epi_coclass_mat[odor_conj_coclass_mat == 1] = 2
    odor_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    odor_core_epi_coclass_mat[odor_specif_coclass_mat == 1] = 3
    
    odor_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_odor_core_coclass.eps')
    
    plot_3D_igraph_int_mat(odor_core_epi_coclass_file,odor_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Blue","Cyan"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
    
    
    ###reseau commun recall_WWW recall_what
    print "Computing conj recall coclass mat"
    
    recall_conj_coclass_mat = np.sum(signif_all_coclass_mat[(2,3),:,:],axis = 0)
    
    recall_conj_coclass_mat[recall_conj_coclass_mat != 2] = 0
    recall_conj_coclass_mat[recall_conj_coclass_mat == 2] = 1
    
    recall_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(recall_conj_coclass_mat)
    
    recall_conj_degree =  np.sum(recall_conj_coclass_mat,axis = 0)
    
    print recall_conj_degree.shape
    
    ### reseau specif recall WWW
    
    print "Computing specif recall WWW coclass mat"
    recall_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    #print recall_specif_coclass_mat
    #print np.sum(recall_specif_coclass_mat)
    
    recall_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    recall_specif_coclass_mat[recall_conj_coclass_mat == 1] = 0
    
    print np.sum(recall_specif_coclass_mat)
    
    
    recall_specif_degree =  np.sum(recall_specif_coclass_mat,axis = 0)
    
    print recall_specif_degree.shape
    
    
    
    node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_recall_labels[4 <= core_degree] = 1
    node_core_recall_labels[4 <= recall_conj_degree] = 2
    node_core_recall_labels[4 <= recall_specif_degree] = 3
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[4 <= core_degree] = core_degree[4 <= core_degree] * 2
    node_core_recall_sizes[4 <= recall_conj_degree] = recall_conj_degree[4 <= recall_conj_degree] * 2
    node_core_recall_sizes[4 <= recall_specif_degree] = recall_specif_degree[4 <= recall_specif_degree] * 2
    
    
    
    
    
    
    
    ######## plotting core + recall
    
    recall_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    recall_core_epi_coclass_mat[recall_conj_coclass_mat == 1] = 2
    recall_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    recall_core_epi_coclass_mat[recall_specif_coclass_mat == 1] = 3
    
    recall_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_recall_core_coclass.eps')
    
    plot_3D_igraph_int_mat(recall_core_epi_coclass_file,recall_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Red","Pink"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
    print "Saving degree"
    
    tab_degree = np.column_stack((core_degree,epi_degree,odor_conj_degree,odor_specif_degree,recall_conj_degree,recall_specif_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Core','Episodic','Conj_odor','Signif_odor','Conj_recall','signif_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_degrees_node.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
        
def gather_coclass_excluded_results2():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    
    ### reseau commum aux 4 conditions
    print "Computing core coclass mat"
    
    core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    core_coclass_mat[core_coclass_mat != 4] = 0
    core_coclass_mat[core_coclass_mat == 4] = 1
    
    print np.sum(core_coclass_mat)
    
    core_degree = np.sum(core_coclass_mat,axis = 0)
    
    print core_degree
    print core_degree.shape
    
    ### reseau commum aux 2 conditions epi
    print "Computing epi coclass mat"
    
    epi_coclass_mat = np.sum(signif_all_coclass_mat[(0,2),:,:],axis = 0)
    
    #print epi_coclass_mat
    
    epi_coclass_mat[epi_coclass_mat != 2] = 0
    epi_coclass_mat[epi_coclass_mat == 2] = 1
    
    epi_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(epi_coclass_mat)
    
    epi_degree =  np.sum(epi_coclass_mat,axis = 0)
    
    print epi_degree.shape
    
    
    
    
    node_core_epi_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_epi_labels[4 <= core_degree] = 1
    node_core_epi_labels[4 <= epi_degree] = 2
    
    print node_core_epi_labels
    
    node_core_epi_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    #node_core_epi_sizes[4 <= core_degree] = core_degree[4 <= core_degree] * 2
    #node_core_epi_sizes[4 <= epi_degree] = epi_degree[4 <= epi_degree] * 2
    
    
    node_core_epi_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    node_core_epi_sizes[epi_degree != 0] = epi_degree[epi_degree != 0] * 2
    
    ######## plotting core + epi
    
    core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    core_epi_coclass_mat[core_coclass_mat == 1] = 1
    
    core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_epi_coclass2.eps')
    
    plot_3D_igraph_int_mat(core_epi_coclass_file,core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black"],node_col_labels = node_core_epi_labels,nodes_sizes = node_core_epi_sizes)
    
    ###reseau commun odor_WWW odor_what
    print "Computing conj odor coclass mat"
    
    odor_conj_coclass_mat = np.sum(signif_all_coclass_mat[(0,1),:,:],axis = 0)
    
    odor_conj_coclass_mat[odor_conj_coclass_mat != 2] = 0
    odor_conj_coclass_mat[odor_conj_coclass_mat == 2] = 1
    
    odor_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(odor_conj_coclass_mat)
    
    odor_conj_degree =  np.sum(odor_conj_coclass_mat,axis = 0)
    
    print odor_conj_degree.shape
    
    ### reseau specif odor WWW
    
    print "Computing specif odor WWW coclass mat"
    odor_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    odor_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    odor_specif_coclass_mat[odor_conj_coclass_mat == 1] = 0
    print np.sum(odor_specif_coclass_mat)
    
    
    odor_specif_degree =  np.sum(odor_specif_coclass_mat,axis = 0)
    
    print odor_specif_degree.shape
            
    node_core_odor_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_odor_labels[core_degree != 0] = 1
    node_core_odor_labels[np.logical_and(odor_conj_degree != 0 ,core_degree <= odor_conj_degree)] = 2
    node_core_odor_labels[np.logical_and(odor_specif_degree != 0, odor_conj_degree <= odor_specif_degree)] = 3
    
    print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[node_core_odor_labels == 1] = core_degree[node_core_odor_labels == 1] * 2
    node_core_odor_sizes[node_core_odor_labels == 2] = odor_conj_degree[node_core_odor_labels == 2] * 2
    node_core_odor_sizes[node_core_odor_labels == 3] = odor_specif_degree[node_core_odor_labels == 3] * 2
    
    
    ######## plotting core + odor
    
    odor_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    odor_core_epi_coclass_mat[odor_conj_coclass_mat == 1] = 2
    odor_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    odor_core_epi_coclass_mat[odor_specif_coclass_mat == 1] = 3
    
    odor_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_odor_core_coclass2.eps')
    
    plot_3D_igraph_int_mat(odor_core_epi_coclass_file,odor_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Blue","Cyan"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
    
    
    ###reseau commun recall_WWW recall_what
    print "Computing conj recall coclass mat"
    
    recall_conj_coclass_mat = np.sum(signif_all_coclass_mat[(2,3),:,:],axis = 0)
    
    recall_conj_coclass_mat[recall_conj_coclass_mat != 2] = 0
    recall_conj_coclass_mat[recall_conj_coclass_mat == 2] = 1
    
    recall_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(recall_conj_coclass_mat)
    
    recall_conj_degree =  np.sum(recall_conj_coclass_mat,axis = 0)
    
    print recall_conj_degree.shape
    
    ### reseau specif recall WWW
    
    print "Computing specif recall WWW coclass mat"
    recall_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    #print recall_specif_coclass_mat
    #print np.sum(recall_specif_coclass_mat)
    
    recall_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    recall_specif_coclass_mat[recall_conj_coclass_mat == 1] = 0
    
    print np.sum(recall_specif_coclass_mat)
    
    
    recall_specif_degree =  np.sum(recall_specif_coclass_mat,axis = 0)
    
    print recall_specif_degree.shape
    
    
    
    #node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    #node_core_recall_labels[4 <= core_degree] = 1
    #node_core_recall_labels[4 <= recall_conj_degree] = 2
    #node_core_recall_labels[4 <= recall_specif_degree] = 3
    
    #print node_core_recall_labels
    
    #node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    
    #node_core_recall_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    #node_core_recall_sizes[recall_conj_degree != 0] = recall_conj_degree[recall_conj_degree != 0] * 2
    #node_core_recall_sizes[recall_specif_degree != 0] = recall_specif_degree[recall_specif_degree != 0] * 2
        
        
        
        
    #print np.argmax(np.vstack((recall_specif_degree,recall_conj_degree,core_degree)),axis = 0) + 1
    
    #node_core_recall_labels = 3 - np.argmax(np.vstack((core_degree,recall_conj_degree,recall_specif_degree)),axis = 0) 
    
    #print node_core_recall_labels
    
    node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_recall_labels[core_degree != 0] = 1
    node_core_recall_labels[np.logical_and(recall_conj_degree != 0 ,core_degree <= recall_conj_degree)] = 2
    node_core_recall_labels[np.logical_and(recall_specif_degree != 0, core_degree <= recall_specif_degree)] = 3
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[node_core_recall_labels == 1] = core_degree[node_core_recall_labels == 1] * 2
    node_core_recall_sizes[node_core_recall_labels == 2] = recall_conj_degree[node_core_recall_labels == 2] * 2
    node_core_recall_sizes[node_core_recall_labels == 3] = recall_specif_degree[node_core_recall_labels == 3] * 2
    
    
        
    ######## plotting core + recall
    
    recall_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    recall_core_epi_coclass_mat[recall_conj_coclass_mat == 1] = 2
    recall_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    recall_core_epi_coclass_mat[recall_specif_coclass_mat == 1] = 3
    
    recall_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_recall_core_coclass2.eps')
    
    plot_3D_igraph_int_mat(recall_core_epi_coclass_file,recall_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Red","Pink"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
    print "Saving degree"
    
    tab_degree = np.column_stack((core_degree,epi_degree,odor_conj_degree,odor_specif_degree,recall_conj_degree,recall_specif_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Core','Episodic','Conj_odor','Signif_odor','Conj_recall','signif_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_degrees_node.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
        
        
def gather_coclass_excluded_results3():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    
    ### reseau commum aux 4 conditions
    print "Computing core coclass mat"
    
    core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    core_coclass_mat[core_coclass_mat != 4] = 0
    core_coclass_mat[core_coclass_mat == 4] = 1
    
    print np.sum(core_coclass_mat)
    
    core_degree = np.sum(core_coclass_mat,axis = 0)
    
    print core_degree
    print core_degree.shape
    
    ### reseau commum aux 2 conditions epi
    print "Computing epi coclass mat"
    
    epi_coclass_mat = np.sum(signif_all_coclass_mat[(0,2),:,:],axis = 0)
    
    #print epi_coclass_mat
    
    epi_coclass_mat[epi_coclass_mat != 2] = 0
    epi_coclass_mat[epi_coclass_mat == 2] = 1
    
    epi_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(epi_coclass_mat)
    
    epi_degree =  np.sum(epi_coclass_mat,axis = 0)
    
    print epi_degree.shape
    
    
    
    
    node_core_epi_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_epi_labels[4 <= core_degree] = 1
    node_core_epi_labels[4 <= epi_degree] = 2
    
    print node_core_epi_labels
    
    node_core_epi_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    #node_core_epi_sizes[4 <= core_degree] = core_degree[4 <= core_degree] * 2
    #node_core_epi_sizes[4 <= epi_degree] = epi_degree[4 <= epi_degree] * 2
    
    
    node_core_epi_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    node_core_epi_sizes[epi_degree != 0] = epi_degree[epi_degree != 0] * 2
    
    ######## plotting core + epi
    
    core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    core_epi_coclass_mat[epi_coclass_mat == 1] = 2
    core_epi_coclass_mat[core_coclass_mat == 1] = 1
    
    core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_epi_coclass3.eps')
    
    plot_3D_igraph_int_mat(core_epi_coclass_file,core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Black"],node_col_labels = node_core_epi_labels,nodes_sizes = node_core_epi_sizes)
    
    ###reseau commun odor_WWW odor_what
    print "Computing conj odor coclass mat"
    
    odor_conj_coclass_mat = np.sum(signif_all_coclass_mat[(0,1),:,:],axis = 0)
    
    odor_conj_coclass_mat[odor_conj_coclass_mat != 2] = 0
    odor_conj_coclass_mat[odor_conj_coclass_mat == 2] = 1
    
    odor_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(odor_conj_coclass_mat)
    
    odor_conj_degree =  np.sum(odor_conj_coclass_mat,axis = 0)
    
    print odor_conj_degree.shape
    
    ### reseau specif odor WWW
    
    print "Computing specif odor WWW coclass mat"
    odor_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    #print odor_specif_coclass_mat
    #print np.sum(odor_specif_coclass_mat)
    
    odor_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    odor_specif_coclass_mat[odor_conj_coclass_mat == 1] = 0
    #odor_specif_coclass_mat[core_coclass_mat == 1] = 0
    
    
    print np.sum(odor_specif_coclass_mat)
    
    
    odor_specif_degree =  np.sum(odor_specif_coclass_mat,axis = 0)
    
    print odor_specif_degree.shape
            
    node_core_odor_labels = 3 - np.argmax(np.vstack((core_degree,odor_conj_degree,odor_specif_degree)),axis = 0) 
    
    #node_core_odor_labels = np.argmax(np.vstack((core_degree,odor_conj_degree,odor_specif_degree)),axis = 0) + 1
    
    print node_core_odor_labels
    
    #node_core_odor_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    #node_core_odor_labels[core_degree != 0] = 1
    #node_core_odor_labels[np.logical_and(odor_conj_degree != 0 ,core_degree <= odor_conj_degree)] = 2
    #node_core_odor_labels[np.logical_and(odor_specif_degree != 0, odor_conj_degree <= odor_specif_degree)] = 3
    
    #print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[node_core_odor_labels == 1] = core_degree[node_core_odor_labels == 1] * 2
    node_core_odor_sizes[node_core_odor_labels == 2] = odor_conj_degree[node_core_odor_labels == 2] * 2
    node_core_odor_sizes[node_core_odor_labels == 3] = odor_specif_degree[node_core_odor_labels == 3] * 2
    
    
    ######## plotting core + odor
    
    odor_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    odor_core_epi_coclass_mat[odor_conj_coclass_mat == 1] = 2
    odor_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    odor_core_epi_coclass_mat[odor_specif_coclass_mat == 1] = 3
    
    odor_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_odor_core_coclass3.eps')
    
    plot_3D_igraph_int_mat(odor_core_epi_coclass_file,odor_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Blue","Cyan"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
    
    
    ###reseau commun recall_WWW recall_what
    print "Computing conj recall coclass mat"
    
    recall_conj_coclass_mat = np.sum(signif_all_coclass_mat[(2,3),:,:],axis = 0)
    
    recall_conj_coclass_mat[recall_conj_coclass_mat != 2] = 0
    recall_conj_coclass_mat[recall_conj_coclass_mat == 2] = 1
    
    recall_conj_coclass_mat[core_coclass_mat == 1] = 0
    
    print np.sum(recall_conj_coclass_mat)
    
    recall_conj_degree =  np.sum(recall_conj_coclass_mat,axis = 0)
    
    print recall_conj_degree.shape
    
    ### reseau specif recall WWW
    
    print "Computing specif recall WWW coclass mat"
    recall_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    
    #print recall_specif_coclass_mat
    #print np.sum(recall_specif_coclass_mat)
    
    recall_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    recall_specif_coclass_mat[recall_conj_coclass_mat == 1] = 0
    
    print np.sum(recall_specif_coclass_mat)
    
    
    recall_specif_degree =  np.sum(recall_specif_coclass_mat,axis = 0)
    
    print recall_specif_degree.shape
    
    
    
    #node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    #node_core_recall_labels[4 <= core_degree] = 1
    #node_core_recall_labels[4 <= recall_conj_degree] = 2
    #node_core_recall_labels[4 <= recall_specif_degree] = 3
    
    #print node_core_recall_labels
    
    #node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    
    #node_core_recall_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    #node_core_recall_sizes[recall_conj_degree != 0] = recall_conj_degree[recall_conj_degree != 0] * 2
    #node_core_recall_sizes[recall_specif_degree != 0] = recall_specif_degree[recall_specif_degree != 0] * 2
        
        
        
        
    print np.argmax(np.vstack((recall_specif_degree,recall_conj_degree,core_degree)),axis = 0) + 1
    
    node_core_recall_labels = 3 - np.argmax(np.vstack((core_degree,recall_conj_degree,recall_specif_degree)),axis = 0) 
    
    print node_core_recall_labels
    
    #node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    #node_core_recall_labels[core_degree != 0] = 1
    #node_core_recall_labels[np.logical_and(recall_conj_degree != 0 ,core_degree <= recall_conj_degree)] = 2
    #node_core_recall_labels[np.logical_and(recall_specif_degree != 0, core_degree <= recall_specif_degree)] = 3
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[node_core_recall_labels == 1] = core_degree[node_core_recall_labels == 1] * 2
    node_core_recall_sizes[node_core_recall_labels == 2] = recall_conj_degree[node_core_recall_labels == 2] * 2
    node_core_recall_sizes[node_core_recall_labels == 3] = recall_specif_degree[node_core_recall_labels == 3] * 2
    
    
        
    ######## plotting core + recall
    
    recall_core_epi_coclass_mat = np.zeros(shape = core_coclass_mat.shape, dtype = int)
    
    recall_core_epi_coclass_mat[recall_conj_coclass_mat == 1] = 2
    recall_core_epi_coclass_mat[core_coclass_mat == 1] = 1
    recall_core_epi_coclass_mat[recall_specif_coclass_mat == 1] = 3
    
    recall_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_recall_core_coclass3.eps')
    
    plot_3D_igraph_int_mat(recall_core_epi_coclass_file,recall_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray","Red","Pink"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
    print "Saving degree"
    
    tab_degree = np.column_stack((core_degree,epi_degree,odor_conj_degree,odor_specif_degree,recall_conj_degree,recall_specif_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Core','Episodic','Conj_odor','Signif_odor','Conj_recall','signif_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_degrees_node.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
        
def gather_coclass_excluded_results4():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    ##################################################### epi core ##################################################
    
    ### reseau commum aux 2 conditions epi
    print "Computing epi coclass mat"
    
    epi_coclass_mat = np.sum(signif_all_coclass_mat[(0,2),:,:],axis = 0)
    
    #print epi_coclass_mat
    
    epi_coclass_mat[epi_coclass_mat != 2] = 0
    epi_coclass_mat[epi_coclass_mat == 2] = 1
    
    
    print np.sum(epi_coclass_mat)
    
    epi_degree =  np.sum(epi_coclass_mat,axis = 0)
    
    print epi_degree.shape
    
    
    
    
    node_core_epi_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_epi_labels[4 <= epi_degree] = 1
    
    print node_core_epi_labels
    
    node_core_epi_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    node_core_epi_sizes[epi_degree != 0] = epi_degree[epi_degree != 0] * 2
    
    ######## plotting core + epi
    
    core_epi_coclass_mat = np.zeros(shape = epi_coclass_mat.shape, dtype = int)
    
    core_epi_coclass_mat[epi_coclass_mat == 1] = 1
    
    core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_epi_coclass4.eps')
    
    plot_3D_igraph_int_mat(core_epi_coclass_file,core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray"],node_col_labels = node_core_epi_labels,nodes_sizes = node_core_epi_sizes)
    
    ################################################## odor ##############################################

        
    ###reseau commun odor_WWW odor_what
    print "Computing conj odor coclass mat"
    
    odor_conj_coclass_mat = np.sum(signif_all_coclass_mat[(0,1),:,:],axis = 0)
    
    odor_conj_coclass_mat[odor_conj_coclass_mat != 2] = 0
    odor_conj_coclass_mat[odor_conj_coclass_mat == 2] = 1
    
    print np.sum(odor_conj_coclass_mat)
    
    odor_conj_degree =  np.sum(odor_conj_coclass_mat,axis = 0)
    
    print odor_conj_degree.shape
    
    ### reseau specif odor WWW
    
    print "Computing specif odor WWW coclass mat"
    odor_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    odor_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    print np.sum(odor_specif_coclass_mat)
    
    odor_specif_degree =  np.sum(odor_specif_coclass_mat,axis = 0)
    
    print odor_specif_degree.shape
            
    node_core_odor_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_odor_labels[odor_conj_degree != 0 ] = 1
    node_core_odor_labels[np.logical_and(odor_specif_degree != 0, odor_conj_degree <= odor_specif_degree)] = 2
    
    print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[node_core_odor_labels == 1] = odor_conj_degree[node_core_odor_labels == 1] * 2
    node_core_odor_sizes[node_core_odor_labels == 2] = odor_specif_degree[node_core_odor_labels == 2] * 2
    
    
    ######## plotting core + odor
    
    odor_core_epi_coclass_mat = np.zeros(shape = epi_coclass_mat.shape, dtype = int)
    
    odor_core_epi_coclass_mat[odor_conj_coclass_mat == 1] = 1
    odor_core_epi_coclass_mat[odor_specif_coclass_mat == 1] = 2
    
    odor_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_odor_core_coclass4.eps')
    
    plot_3D_igraph_int_mat(odor_core_epi_coclass_file,odor_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Orange","Blue"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
    
        
    ################################################## recall ##############################################

        
    ###reseau commun recall_WWW recall_what
    print "Computing conj recall coclass mat"
    
    recall_conj_coclass_mat = np.sum(signif_all_coclass_mat[(2,3),:,:],axis = 0)
    
    recall_conj_coclass_mat[recall_conj_coclass_mat != 2] = 0
    recall_conj_coclass_mat[recall_conj_coclass_mat == 2] = 1
    
    print np.sum(recall_conj_coclass_mat)
    
    recall_conj_degree =  np.sum(recall_conj_coclass_mat,axis = 0)
    
    print recall_conj_degree.shape
    
    ### reseau specif recall WWW
    
    print "Computing specif recall WWW coclass mat"
    recall_specif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    recall_specif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    print np.sum(recall_specif_coclass_mat)
    
    recall_specif_degree =  np.sum(recall_specif_coclass_mat,axis = 0)
    
    print recall_specif_degree.shape
            
    node_core_recall_labels = np.zeros(shape = epi_degree.shape,dtype = int)
    
    node_core_recall_labels[recall_conj_degree != 0 ] = 1
    node_core_recall_labels[np.logical_and(recall_specif_degree != 0, recall_conj_degree <= recall_specif_degree)] = 2
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = epi_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[node_core_recall_labels == 1] = recall_conj_degree[node_core_recall_labels == 1] * 2
    node_core_recall_sizes[node_core_recall_labels == 2] = recall_specif_degree[node_core_recall_labels == 2] * 2
    
    
    ######## plotting core + recall
    
    recall_core_epi_coclass_mat = np.zeros(shape = epi_coclass_mat.shape, dtype = int)
    
    recall_core_epi_coclass_mat[recall_conj_coclass_mat == 1] = 1
    recall_core_epi_coclass_mat[recall_specif_coclass_mat == 1] = 2
    
    recall_core_epi_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_recall_core_coclass4.eps')
    
    plot_3D_igraph_int_mat(recall_core_epi_coclass_file,recall_core_epi_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Green","Red"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
    
    
    print "Saving degree"
    
    tab_degree = np.column_stack((epi_degree,odor_conj_degree,odor_specif_degree,recall_conj_degree,recall_specif_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Episodic','Conj_odor','Signif_odor','Conj_recall','signif_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'degrees_node.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
        
def gather_coclass_excluded_results5():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    ##################################################### core ##################################################
    
    ### reseau commum aux 4 conditions
    print "Computing core coclass mat"
    
    core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    core_coclass_mat[core_coclass_mat != 4] = 0
    core_coclass_mat[core_coclass_mat == 4] = 1
    
    print np.sum(core_coclass_mat)
    
    core_degree = np.sum(core_coclass_mat,axis = 0)
    
    print core_degree
    print core_degree.shape
    
    node_core_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_labels[core_degree != 0] = 1
    
    print node_core_labels
    
    node_core_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    
    core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_coclass5.eps')
    
    plot_3D_igraph_int_mat(core_coclass_file,core_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray"],node_col_labels = node_core_labels,nodes_sizes = node_core_sizes)
    
    
    core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(core_coclass_topo_file,core_coclass_mat, labels = labels, edge_colors = ["Gray"],node_col_labels = node_core_labels,nodes_sizes = node_core_sizes)
    
    
    ################################################## odor ##############################################

    ### reseau signif odor WWW
    
    print "Computing signif odor WWW coclass mat"
    odor_signif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    odor_signif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 25)] = 1
    
    print np.sum(odor_signif_coclass_mat)
    
    odor_signif_degree =  np.sum(odor_signif_coclass_mat,axis = 0)
    
    print odor_signif_degree.shape
            
    node_core_odor_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_odor_labels[odor_signif_degree != 0] = 1
    
    print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[node_core_odor_labels == 1] = odor_signif_degree[node_core_odor_labels == 1] * 2
    
    
    odor_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_odor_core_coclass5.eps')
    
    plot_3D_igraph_int_mat(odor_core_coclass_file,odor_signif_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Blue"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
        
    odor_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_odor_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(odor_core_coclass_topo_file,odor_signif_coclass_mat, labels = labels, edge_colors = ["Blue"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
        
    ################################################## recall ##############################################

        
    ### reseau signif recall WWW
    
    print "Computing signif recall WWW coclass mat"
    recall_signif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    recall_signif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 25)] = 1
    
    print np.sum(recall_signif_coclass_mat)
    
    recall_signif_degree =  np.sum(recall_signif_coclass_mat,axis = 0)
    
    print recall_signif_degree.shape
            
    node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_recall_labels[recall_signif_degree != 0] = 1
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[node_core_recall_labels == 1] = recall_signif_degree[node_core_recall_labels == 1] * 2
    
    
    recall_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_recall_core_coclass5.eps')
    
    plot_3D_igraph_int_mat(recall_core_coclass_file,recall_signif_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Red"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
    recall_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_recall_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(recall_core_coclass_topo_file,recall_signif_coclass_mat, labels = labels, edge_colors = ["Red"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
        
    print "Computing conj signif odor and recall WWW coclass mat"
    
    conj_signif_odor_recall_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    conj_signif_odor_recall_coclass_mat[np.logical_and(recall_signif_coclass_mat == 1, odor_signif_coclass_mat == 1)] = 1
    
    print np.sum(recall_signif_coclass_mat)
    
    conj_signif_odor_recall_degree =  np.sum(conj_signif_odor_recall_coclass_mat,axis = 0)
    
    print conj_signif_odor_recall_degree.shape
            
    conj_signif_odor_recall_labels = np.zeros(shape = conj_signif_odor_recall_degree.shape,dtype = int)
    
    conj_signif_odor_recall_labels[conj_signif_odor_recall_degree != 0] = 1
    
    print conj_signif_odor_recall_labels
    
    conj_signif_odor_recall_sizes = np.ones(shape = conj_signif_odor_recall_degree.shape,dtype = float) *0.1
    
    conj_signif_odor_recall_sizes[conj_signif_odor_recall_labels == 1] = conj_signif_odor_recall_degree[conj_signif_odor_recall_labels == 1] * 2
    
    conj_signif_odor_recall_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass5.eps')
    
    plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_file,conj_signif_odor_recall_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Purple"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
    conj_signif_odor_recall_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_topo_file,conj_signif_odor_recall_coclass_mat, labels = labels, edge_colors = ["Purple"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
    
    
    
    ######## subgraph core for all conditions
    
    core_nodes = core_degree != 0
    
    print core_nodes
    
    subgraph_labels = [label for i,label in enumerate(labels) if core_nodes[i] == True]
    print subgraph_labels
    
    subgraph_MNI_coords = MNI_coords[core_nodes,:]
    print subgraph_MNI_coords
    
    
    
    
    subgraph_core_coclass_mat = core_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph_core_coclass_mat
    
    subgraph_odor_signif_coclass_mat = odor_signif_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph_odor_signif_coclass_mat
    
    subgraph_recall_signif_coclass_mat = recall_signif_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph_recall_signif_coclass_mat
    
    subgraph_conj_signif_odor_recall_coclass_mat = conj_signif_odor_recall_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph_conj_signif_odor_recall_coclass_mat
    
    subgraph_coclass_mat = np.zeros(shape = subgraph_core_coclass_mat.shape, dtype = int)
    
    subgraph_coclass_mat[subgraph_core_coclass_mat == 1] = 1
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 0,subgraph_odor_signif_coclass_mat == 1)] = 2
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 0,subgraph_recall_signif_coclass_mat == 1)] = 3
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 0,subgraph_conj_signif_odor_recall_coclass_mat == 1)] = 4
    
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 1,subgraph_odor_signif_coclass_mat == 1)] = 5
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 1,subgraph_recall_signif_coclass_mat == 1)] = 6
    
    subgraph_coclass_mat[np.logical_and(subgraph_core_coclass_mat == 1,subgraph_conj_signif_odor_recall_coclass_mat == 1)] = 7
    
    
    print subgraph_coclass_mat
    
    subgraph_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'subgraph_core_coclass5_from_top.eps')
    
    plot_3D_igraph_int_mat(subgraph_core_coclass_file,subgraph_coclass_mat, labels = subgraph_labels, coords = subgraph_MNI_coords, edge_colors = ["Gray","Blue","Red","Purple","Green","Orange","Black"], view_from = '_from_top')
    
    subgraph_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'subgraph_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(subgraph_core_coclass_topo_file,subgraph_coclass_mat, labels = subgraph_labels, edge_colors = ["Gray","Blue","Red","Purple","Green","Orange","Black"])
    
    #conj_signif_odor_recall_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass5_topo.eps')
    
    #plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_topo_file,conj_signif_odor_recall_coclass_mat, labels = labels, edge_colors = ["Purple"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    ######## subgraph2 core for all conditions
    
    core_nodes = core_degree != 0
    
    print core_nodes
    
    for keep_label in ['pHip.L', 'aHip.R', 'aPHP/Fus.R', 'pPHC.R','pPir.R', 'pOFC.R','aPir/pOFC.L']:
        
        if keep_label in labels:
            
            index_sub_node = labels.index(keep_label)
            
            print index_sub_node
            
        else:
            print "Warning, could not find %s in labels"%keep_label
            print labels
            
            return
    0/0
    
    subgraph2_labels = [label for i,label in enumerate(labels) if core_nodes[i] == True]
    print subgraph2_labels
    
    subgraph2_MNI_coords = MNI_coords[core_nodes,:]
    print subgraph2_MNI_coords
    
    
    
    
    subgraph2_core_coclass_mat = core_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph2_core_coclass_mat
    
    subgraph2_odor_signif_coclass_mat = odor_signif_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph2_odor_signif_coclass_mat
    
    subgraph2_recall_signif_coclass_mat = recall_signif_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph2_recall_signif_coclass_mat
    
    subgraph2_conj_signif_odor_recall_coclass_mat = conj_signif_odor_recall_coclass_mat[core_nodes,:][:,core_nodes]
    print subgraph2_conj_signif_odor_recall_coclass_mat
    
    subgraph2_coclass_mat = np.zeros(shape = subgraph2_core_coclass_mat.shape, dtype = int)
    
    subgraph2_coclass_mat[subgraph2_core_coclass_mat == 1] = 1
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 0,subgraph2_odor_signif_coclass_mat == 1)] = 2
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 0,subgraph2_recall_signif_coclass_mat == 1)] = 3
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 0,subgraph2_conj_signif_odor_recall_coclass_mat == 1)] = 4
    
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 1,subgraph2_odor_signif_coclass_mat == 1)] = 5
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 1,subgraph2_recall_signif_coclass_mat == 1)] = 6
    
    subgraph2_coclass_mat[np.logical_and(subgraph2_core_coclass_mat == 1,subgraph2_conj_signif_odor_recall_coclass_mat == 1)] = 7
    
    
    print subgraph2_coclass_mat
    
    subgraph2_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'subgraph2_core_coclass5_from_top.eps')
    
    plot_3D_igraph_int_mat(subgraph2_core_coclass_file,subgraph2_coclass_mat, labels = subgraph2_labels, coords = subgraph2_MNI_coords, edge_colors = ["Gray","Blue","Red","Purple","Green","Orange","Black"], view_from = '_from_top')
    
    subgraph2_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'subgraph2_core_coclass5_topo.eps')
    
    plot_3D_igraph_int_mat(subgraph2_core_coclass_topo_file,subgraph2_coclass_mat, labels = subgraph2_labels, edge_colors = ["Gray","Blue","Red","Purple","Green","Orange","Black"])
    
    #conj_signif_odor_recall_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass5_topo.eps')
    
    #plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_topo_file,conj_signif_odor_recall_coclass_mat, labels = labels, edge_colors = ["Purple"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print "Saving degree"
    
    tab_degree = np.column_stack((core_degree,odor_signif_degree,recall_signif_degree,conj_signif_odor_recall_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Core','Signif_odor','signif_recall','conj_signif_odor_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'degrees_node5.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
    
def gather_coclass_excluded_results6():

    from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes_and_sparse_matrix
    from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
    
    from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
    import pandas as pd
    
    ### labels
    
    print 'loading labels'
    
    labels = [line.strip() for line in open(ROI_coords_labels_file)]

    print labels
        
    print 'loading ROI coords'
    
    MNI_coords = np.array(np.loadtxt(ROI_coords_MNI_coords_file),dtype = 'float')
    
    print MNI_coords.shape
        
    list_coclass_mat = []
    
    for cond in ['Odor_Hit-WWW','Odor_Hit-What','Recall_Hit-WWW','Recall_Hit-What']:
    
        coclass_list_file = os.path.join(nipype_analyses_path,coclass_analysis_name,"_cond_" + cond, "prep_rada","int_List.net")
        
        print coclass_list_file
        
        node_corres, sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(coclass_list_file)
        
        print node_corres
        print sparse_mat
        
        gm_coords = np.loadtxt(ROI_coords_MNI_coords_file)
        
        node_coords = gm_coords[node_corres,:]
        
        full_cormat,possible_edge_mat = return_corres_correl_mat(sparse_mat.todense(),node_coords,gm_coords)
        
        print full_cormat.shape
        
        #full_cormat[full_cormat > 0] = 1
        
        list_coclass_mat.append(full_cormat)
        
    print list_coclass_mat
    
    all_coclass_mat = np.array(list_coclass_mat,dtype = int)
    
    print all_coclass_mat
    print all_coclass_mat.shape
    
    signif_all_coclass_mat = np.copy(all_coclass_mat)
    signif_all_coclass_mat[signif_all_coclass_mat > 0] = 1    
    
    print signif_all_coclass_mat
    print signif_all_coclass_mat.shape
    
    print "computing degree"
    
    degree_all_coclass = np.sum(signif_all_coclass_mat,axis = 1)
    
    print degree_all_coclass
    print degree_all_coclass.shape
    
    ##################################################### core ##################################################
    
    ### reseau commum aux 4 conditions
    print "Computing core coclass mat"
    
    core_coclass_mat = np.sum(signif_all_coclass_mat,axis = 0)
    
    core_coclass_mat[core_coclass_mat != 4] = 0
    core_coclass_mat[core_coclass_mat == 4] = 1
    
    print np.sum(core_coclass_mat)
    
    core_degree = np.sum(core_coclass_mat,axis = 0)
    
    print core_degree
    print core_degree.shape
    
    node_core_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_labels[core_degree != 0] = 1
    
    print node_core_labels
    
    node_core_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_sizes[core_degree != 0] = core_degree[core_degree != 0] * 2
    
    core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_coclass6.eps')
    
    plot_3D_igraph_int_mat(core_coclass_file,core_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Gray"],node_col_labels = node_core_labels,nodes_sizes = node_core_sizes)
    
    
    core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'exclu_core_coclass6_topo.eps')
    
    plot_3D_igraph_int_mat(core_coclass_topo_file,core_coclass_mat, labels = labels, edge_colors = ["Gray"],node_col_labels = node_core_labels,nodes_sizes = node_core_sizes)
    
    
    ################################################## odor ##############################################

    ### reseau signif odor WWW
    
    print "Computing signif odor WWW coclass mat"
    odor_signif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    odor_signif_coclass_mat[np.logical_and(signif_all_coclass_mat[0,:,:] == 1, all_coclass_mat[0,:,:] - all_coclass_mat[1,:,:] > 26)] = 1
    odor_signif_coclass_mat[core_coclass_mat == 1] = 2
    
    print np.sum(odor_signif_coclass_mat)
    
    odor_signif_degree =  np.sum(odor_signif_coclass_mat,axis = 0)
    
    print odor_signif_degree.shape
            
    node_core_odor_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_odor_labels[odor_signif_degree != 0] = 1
    node_core_odor_labels[core_degree != 0] = 2
    
    print node_core_odor_labels
    
    node_core_odor_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_odor_sizes[node_core_odor_labels == 1] = odor_signif_degree[node_core_odor_labels == 1] * 2
    node_core_odor_sizes[node_core_odor_labels == 2] = core_degree[node_core_odor_labels == 2] * 2
    
    odor_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_odor_core_coclass6.eps')
    
    plot_3D_igraph_int_mat(odor_core_coclass_file,odor_signif_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Blue","Gray"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
        
    odor_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_odor_core_coclass6_topo.eps')
    
    plot_3D_igraph_int_mat(odor_core_coclass_topo_file,odor_signif_coclass_mat, labels = labels, edge_colors = ["Blue","Gray"],node_col_labels = node_core_odor_labels,nodes_sizes = node_core_odor_sizes)
    ################################################## recall ##############################################

        
    ### reseau signif recall WWW
    
    print "Computing signif recall WWW coclass mat"
    recall_signif_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    recall_signif_coclass_mat[np.logical_and(signif_all_coclass_mat[2,:,:] == 1, all_coclass_mat[2,:,:] - all_coclass_mat[3,:,:] > 26)] = 1
    recall_signif_coclass_mat[core_coclass_mat == 1] = 2
    
    print np.sum(recall_signif_coclass_mat)
    
    recall_signif_degree =  np.sum(recall_signif_coclass_mat,axis = 0)
    
    print recall_signif_degree.shape
            
    node_core_recall_labels = np.zeros(shape = core_degree.shape,dtype = int)
    
    node_core_recall_labels[recall_signif_degree != 0] = 1
    node_core_recall_labels[core_degree != 0] = 2
    
    print node_core_recall_labels
    
    node_core_recall_sizes = np.ones(shape = core_degree.shape,dtype = float) *0.1
    
    node_core_recall_sizes[node_core_recall_labels == 1] = recall_signif_degree[node_core_recall_labels == 1] * 2
    node_core_recall_sizes[node_core_recall_labels == 2] = core_degree[node_core_recall_labels == 2] * 2
    
    recall_core_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_recall_core_coclass6.eps')
    
    plot_3D_igraph_int_mat(recall_core_coclass_file,recall_signif_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Red","Gray"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
        
    recall_core_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'signif_recall_core_coclass6_topo.eps')
    
    plot_3D_igraph_int_mat(recall_core_coclass_topo_file,recall_signif_coclass_mat, labels = labels, edge_colors = ["Red","Gray"],node_col_labels = node_core_recall_labels,nodes_sizes = node_core_recall_sizes)
    
        
    print "Computing conj signif odor and recall WWW coclass mat"
    
    conj_signif_odor_recall_coclass_mat = np.zeros(shape = signif_all_coclass_mat.shape[1:],dtype = int)
    
    conj_signif_odor_recall_coclass_mat[np.logical_and(recall_signif_coclass_mat == 1, odor_signif_coclass_mat == 1)] = 1
    conj_signif_odor_recall_coclass_mat[core_coclass_mat == 1] = 2
    
    print np.sum(recall_signif_coclass_mat)
    
    conj_signif_odor_recall_degree =  np.sum(conj_signif_odor_recall_coclass_mat,axis = 0)
    
    print conj_signif_odor_recall_degree.shape
            
    conj_signif_odor_recall_labels = np.zeros(shape = conj_signif_odor_recall_degree.shape,dtype = int)
    
    conj_signif_odor_recall_labels[conj_signif_odor_recall_degree != 0] = 1
    conj_signif_odor_recall_labels[core_degree != 0] = 2
    
    print conj_signif_odor_recall_labels
    
    conj_signif_odor_recall_sizes = np.ones(shape = conj_signif_odor_recall_degree.shape,dtype = float) *0.1
    
    conj_signif_odor_recall_sizes[conj_signif_odor_recall_labels == 1] = conj_signif_odor_recall_degree[conj_signif_odor_recall_labels == 1] * 2
    conj_signif_odor_recall_sizes[conj_signif_odor_recall_labels == 2] = core_degree[conj_signif_odor_recall_labels == 2] * 2
    
    
    conj_signif_odor_recall_coclass_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass6.eps')
    
    plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_file,conj_signif_odor_recall_coclass_mat, labels = labels, coords = MNI_coords, edge_colors = ["Purple","Gray"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
    conj_signif_odor_recall_coclass_topo_file = os.path.join(nipype_analyses_path,coclass_analysis_name,'conj_signif_odor_recall_core_coclass6_topo.eps')
    
    plot_3D_igraph_int_mat(conj_signif_odor_recall_coclass_topo_file,conj_signif_odor_recall_coclass_mat, labels = labels, edge_colors = ["Purple","Gray"],node_col_labels = conj_signif_odor_recall_labels,nodes_sizes = conj_signif_odor_recall_sizes)
    
    print "Saving degree"
    
    tab_degree = np.column_stack((core_degree,odor_signif_degree,recall_signif_degree,conj_signif_odor_recall_degree))
    
    df = pd.DataFrame(tab_degree,columns = ['Core','Signif_odor','signif_recall','conj_signif_odor_recall'],index = labels)
    
    df_filename = os.path.join(nipype_analyses_path,coclass_analysis_name,'degrees_node6.txt')
    
    df.to_csv(df_filename)
    
    print tab_degree.shape
        
        
        
if __name__ =='__main__':
    
    ########### testing colors
    #test_generate_igraph_colors()
    
    ############ run all 
    #print split_coclass_analysis_name

    #if 'diff' in split_coclass_analysis_name:
    
        #if 'cond' in split_coclass_analysis_name:
                    
            #print "run diff_cond"
            #run_coclass_diff_cond()
            
        #elif 'event' in split_coclass_analysis_name:
                    
            #print "run diff_event"
            #run_coclass_diff_event()
        
    #else:
        #print "run coclass by cond"
        #run_coclass_by_cond()
    
    
    ############ gather results
    #gather_coclass_results()
    #gather_coclass_excluded_results()
    #gather_coclass_excluded_results2()
    #gather_coclass_excluded_results3()
    #gather_coclass_excluded_results4()
    
    gather_coclass_excluded_results5()
    
    #gather_coclass_excluded_results6()
    