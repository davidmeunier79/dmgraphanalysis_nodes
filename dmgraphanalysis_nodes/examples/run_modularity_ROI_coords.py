# -*- coding: utf-8 -*-
"""
F) Compute weighted signed modularity from weighted correlation matrices
"""

import sys, os
sys.path.append('../irm_analysis')

#from nipype import config
#config.enable_debug_mode()

#import sys,io,os
import rpy

from  define_variables import *

#### loop all set of data

import nitime.algorithms.correlation as cor 

import scipy.io
from scipy import stats

from dmgraphanalysis_nodes.nodes.modularity import ComputeNetList,PrepRada,CommRada
from dmgraphanalysis_nodes.nodes.modularity import PlotIGraphModules ## maybe could be in a special iGraph plot interface

################################################ Infosource/Datasource

def create_inforsource():
    
    infosource = pe.Node(interface=IdentityInterface(fields=['subject_num', 'cond']),name="infosource")
    
    ### all subjects in one go
    infosource.iterables = [('subject_num', subject_nums),('cond',epi_cond)]
    
    ## testing
    #infosource.iterables = [('subject_num', ['S08']),('cond',['Odor_Hit-WWW'])]
    
    return infosource
    
def create_datasource_conf_correl_mat():
    
    datasource_preproc = pe.Node(interface=nio.DataGrabber(infields=['subject_num','cond'],outfields=['cor_mat_file','conf_cor_mat_file','coords_file','resid_ts_file','regressor_file']),name = 'datasource_preproc')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource_preproc.inputs.base_directory = os.path.join(nipype_analyses_path,cor_mat_analysis_name)
    datasource_preproc.inputs.template = '_cond_%s_subject_num_%s/%s/%s'

    datasource_preproc.inputs.template_args = dict(
        conf_cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","conf_cor_mat.npy"]],
        cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","cor_mat.npy"]],
        coords_file= [['cond','subject_num',"merge_runs","coord_rois_all_runs.txt"]],
        resid_ts_file= [['cond','subject_num',"merge_runs","ts_all_runs.npy"]],
        regressor_file = [['cond','subject_num',"merge_runs","regressor_all_runs_file.txt"]]
        )

    
    datasource_preproc.inputs.sort_filelist = True
    
    return datasource_preproc
    

def create_datasource_Z_correl_mat():
    
    #### Data source from Z correlations
    datasource_preproc = pe.Node(interface=nio.DataGrabber(infields=['subject_num','cond'],outfields=['Z_cor_mat_file','coords_file','resid_ts_file','regressor_file']),name = 'datasource_preproc')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource_preproc.inputs.base_directory = os.path.join(nipype_analyses_path,cor_mat_analysis_name)
    datasource_preproc.inputs.template = '_cond_%s_subject_num_%s/%s/%s'

    datasource_preproc.inputs.template_args = dict(
        Z_cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","Z_cor_mat.npy"]],
        coords_file= [['cond','subject_num',"merge_runs","coord_rois_all_runs.txt"]],
        resid_ts_file= [['cond','subject_num',"merge_runs","ts_all_runs.npy"]],
        regressor_file = [['cond','subject_num',"merge_runs","regressor_all_runs.txt"]]
        )

    datasource_preproc.inputs.sort_filelist = True
    
    return datasource_preproc
    
#################################################### Main Workflow #################################################################
    
def create_wei_sig_modularity_workflow():
    
    
    main_workflow = Workflow(name=graph_analysis_name)
    main_workflow.base_dir = nipype_analyses_path
    
    #### Info source
    infosource = create_inforsource()
    

    datasource_preproc = create_datasource_Z_correl_mat()

    main_workflow.connect(infosource, 'subject_num', datasource_preproc, 'subject_num')
    main_workflow.connect(infosource, 'cond', datasource_preproc, 'cond')
    
    compute_net_List = pe.Node(interface = ComputeNetList(),name='compute_net_List')
    
    main_workflow.connect(datasource_preproc, 'Z_cor_mat_file',compute_net_List, 'Z_cor_mat_file')
    main_workflow.connect(datasource_preproc, 'coords_file',compute_net_List, 'coords_file')
    
    #################################################### radatools ################################################################

    ### prepare net_list for radatools processing  
    prep_rada = pe.Node(interface = PrepRada(),name='prep_rada')
    #Function(input_names=['List_net_file','radatools_prep_path'],output_names = ['Pajek_net_file'],function = prep_radatools),name='prep_rada')
    prep_rada.inputs.radatools_path = radatools_path
    
    main_workflow.connect(compute_net_List, 'net_List_file', prep_rada, 'net_List_file')
    
    ### compute community with radatools
    community_rada = pe.Node(interface = CommRada(), name='community_rada')
    community_rada.inputs.optim_seq = radatools_optim
    community_rada.inputs.radatools_path = radatools_path
    
    main_workflow.connect( prep_rada, 'Pajek_net_file',community_rada,'Pajek_net_file')
    

        #### read lol file and export modules as img file
        #export_lol_mask = pe.Node(Function(input_names=['rada_lol_file','Pajek_net_file','coords_file','mask_file'],output_names = ['lol_mask_file'],function = export_lol_mask_file),name='export_lol_mask')
        #export_lol_mask.inputs.mask_file = ROI_coords_labelled_mask_file
    
        #main_workflow.connect(community_rada, 'rada_lol_file',export_lol_mask,'rada_lol_file')
        #main_workflow.connect(prep_rada, 'Pajek_net_file',export_lol_mask,'Pajek_net_file')
        #main_workflow.connect(datasource_preproc, 'coords_file',export_lol_mask,'coords_file')
        
        
        #### compute average time series for each module
        #mod_ts_rada = pe.Node(Function(input_names=['ts_mat_file','coords_file','rada_lol_file','Pajek_net_file'],output_names = ['mod_average_ts_file','mod_average_coords_file'],function = compute_mod_average_ts_rada),name='mod_ts_rada')
        
        #main_workflow.connect(datasource_preproc, 'coords_file',mod_ts_rada,'coords_file')
        #main_workflow.connect(datasource_preproc, 'resid_ts_file', mod_ts_rada, 'ts_mat_file')
        #main_workflow.connect(community_rada, 'rada_lol_file',mod_ts_rada,'rada_lol_file')
        #main_workflow.connect(prep_rada, 'Pajek_net_file',mod_ts_rada,'Pajek_net_file')
        
        
        #### compute weighted correlations for modular average time series
        #mod_cor_mat_rada = pe.Node(Function(input_names=['mod_average_ts_file','regressor_file'],output_names = ['mod_cor_mat_file','mod_Z_cor_mat_file'],function = compute_mod_cor_mat),name='mod_cor_mat_rada')
        
        #main_workflow.connect(mod_ts_rada, 'mod_average_ts_file', mod_cor_mat_rada, 'mod_average_ts_file')
        #main_workflow.connect(datasource_preproc, 'regressor_file', mod_cor_mat_rada, 'regressor_file')
        
        
        #### plot igraph matrix
        #plot_igraph_rada = pe.Node(Function(input_names=['mod_cor_mat_file','mod_average_coords_file'],output_names = ['igraph_file'],function = plot_igraph_matrix),name='plot_igraph_rada')
        
        #main_workflow.connect(mod_cor_mat_rada, 'mod_cor_mat_file', plot_igraph_rada, 'mod_cor_mat_file')
        #main_workflow.connect(mod_ts_rada, 'mod_average_coords_file', plot_igraph_rada, 'mod_average_coords_file')
    
    
    #### plot_igraph_modules_rada
    
    plot_igraph_modules_rada = pe.Node(interface = PlotIGraphModules(),name='plot_igraph_modules_rada')
    
    #Function(input_names=['rada_lol_file','Pajek_net_file','coords_file'],output_names = ['Z_list_single_modules_files,Z_list_all_modules_files'],function = plot_igraph_modules_conf_cor_mat_rada),name='plot_igraph_modules_rada')
    plot_igraph_modules_rada.inputs.labels_file = ROI_coords_labels_file

    main_workflow.connect(prep_rada, 'Pajek_net_file',plot_igraph_modules_rada,'Pajek_net_file')
    main_workflow.connect(community_rada, 'rada_lol_file',plot_igraph_modules_rada,'rada_lol_file')
    
    main_workflow.connect(datasource_preproc, 'coords_file',plot_igraph_modules_rada,'coords_file')
    #main_workflow.connect(compute_net_List, 'net_List_file', plot_igraph_modules_rada, 'net_List_file')
    
    
        
    #### plot dist matrix 
    #plot_dist = pe.Node(Function(input_names=['dist_mat_file'],output_names = ['plot_hist_dist_mat_file','plot_heatmap_dist_mat_file'],function = plot_dist_matrix),name='plot_dist')
        
    #main_workflow.connect(compute_net_List, 'dist_mat_file', plot_dist, 'dist_mat_file')
        
    return main_workflow
    
                
def gather_modularity_values():

    import numpy as np
    import pandas as pd
    
    mod_values = []
    for cond in epi_cond:
    
        print cond
    
        mod_cond_values = []
        for  subject_num in subject_nums:
            
            print subject_num
            
            modularity_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"community_rada","Z_List.lol")
            
            print modularity_file
            
            mod_val = get_modularity_value_from_lol_file(modularity_file)
            
            print mod_val
            
            mod_cond_values.append(mod_val)
            
        mod_values.append(mod_cond_values)
        
    print mod_values
    
    np_mod_values = np.array(mod_values,dtype = 'f')
    
    print np_mod_values.shape
    
    df = pd.DataFrame(np.transpose(np_mod_values),columns = epi_cond,index = subject_nums)
    
    df_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'mod_values_by_cond.txt')
    
    df.to_csv(df_filename)
    
            
if __name__ =='__main__':
    
    
    #### compute modular decomposition
    print split_graph_analysis_name
        
    main_workflow = create_wei_sig_modularity_workflow()
    

    #main_workflow.write_graph(analysis_name +'graph.dot',graph2use='flat', format = 'svg')    
    main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 1})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
    main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})

    #### gathering results
    #gather_modularity_values()
    