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

from dmgraphanalysis_nodes.nodes.modularity import ComputeNetList,PrepRada,CommRada,NetPropRada
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
    datasource_preproc.inputs.template = '_cond_%s_subject_num_%s/%s/%s*%s'

    datasource_preproc.inputs.template_args = dict(
        conf_cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","conf_cor_mat",".npy"]],
        cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","cor_mat",".npy"]],
        coords_file= [['cond','subject_num',"merge_runs","","coord_rois_all_runs.txt"]],
        resid_ts_file= [['cond','subject_num',"merge_runs","","ts_all_runs.npy"]],
        regressor_file = [['cond','subject_num',"merge_runs","","regressor_all_runs_file.txt"]]
        )

    
    datasource_preproc.inputs.sort_filelist = True
    
    return datasource_preproc
    

def create_datasource_Z_correl_mat():
    
    #### Data source from Z correlations
    datasource_preproc = pe.Node(interface=nio.DataGrabber(infields=['subject_num','cond'],outfields=['Z_cor_mat_file','coords_file','resid_ts_file','regressor_file']),name = 'datasource_preproc')
    #datasource_preproc.inputs.base_directory = change_name_nifti_path
    datasource_preproc.inputs.base_directory = os.path.join(nipype_analyses_path,cor_mat_analysis_name)
    datasource_preproc.inputs.template = '_cond_%s_subject_num_%s/%s/%s*%s'

    datasource_preproc.inputs.template_args = dict(
        Z_cor_mat_file=[['cond','subject_num',"compute_conf_cor_mat","Z_cor_mat",".npy"]],
        coords_file= [['cond','subject_num',"merge_runs","","coord_rois_all_runs.txt"]],
        resid_ts_file= [['cond','subject_num',"merge_runs","","ts_all_runs.npy"]],
        regressor_file = [['cond','subject_num',"merge_runs","","regressor_all_runs.txt"]]
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
    
    #### plot_igraph_modules_rada
    plot_igraph_modules_rada = pe.Node(interface = PlotIGraphModules(),name='plot_igraph_modules_rada')
    
    #Function(input_names=['rada_lol_file','Pajek_net_file','coords_file'],output_names = ['Z_list_single_modules_files,Z_list_all_modules_files'],function = plot_igraph_modules_conf_cor_mat_rada),name='plot_igraph_modules_rada')
    plot_igraph_modules_rada.inputs.labels_file = ROI_coords_labels_file

    main_workflow.connect(prep_rada, 'Pajek_net_file',plot_igraph_modules_rada,'Pajek_net_file')
    main_workflow.connect(community_rada, 'rada_lol_file',plot_igraph_modules_rada,'rada_lol_file')
    
    main_workflow.connect(datasource_preproc, 'coords_file',plot_igraph_modules_rada,'coords_file')
    #main_workflow.connect(compute_net_List, 'net_List_file', plot_igraph_modules_rada, 'net_List_file')
    
    ############ compute network properties with rada
    net_prop = pe.Node(interface = NetPropRada(optim_seq = "A"), name = 'net_prop')
    net_prop.inputs.radatools_path = radatools_path
    
    main_workflow.connect(prep_rada, 'Pajek_net_file',net_prop,'Pajek_net_file')
    
    
    #### plot dist matrix 
    #plot_dist = pe.Node(Function(input_names=['dist_mat_file'],output_names = ['plot_hist_dist_mat_file','plot_heatmap_dist_mat_file'],function = plot_dist_matrix),name='plot_dist')
        
    #main_workflow.connect(compute_net_List, 'dist_mat_file', plot_dist, 'dist_mat_file')
        
    return main_workflow
    
                
def gather_modularity_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_mod import get_modularity_value_from_lol_file
    
    
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
    
def gather_strength_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_net import get_strength_values_from_info_nodes_file
    
    node_labels = np.array([line.strip() for line in open(ROI_coords_labels_file)],dtype = 'str')
    
    print node_labels
        
    strength_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'strength_values_by_cond.xls')
    
    strength_writer = pd.ExcelWriter(strength_filename)
    
    for cond in epi_cond:
    
        print cond
    
        strength_cond_values = []
        
        for  subject_num in subject_nums:
            
            print subject_num
            
            node_prop_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"net_prop","Z_List-info_nodes.txt")
            
            print node_prop_file
            
            #0/0
            
            node_strength_values = get_strength_values_from_info_nodes_file(node_prop_file)
            
            print node_strength_values
            
            
            strength_cond_values.append(node_strength_values)
            
            #print eff_val
            
            #eff_cond_values.append(eff_val)
            
        np_strength_cond_values = np.array(strength_cond_values,dtype = 'int64')
        
        print np_strength_cond_values
        
        
        df = pd.DataFrame(np_strength_cond_values,columns = node_labels,index = subject_nums, dtype = 'int64')
        
        print df
        
        print df.columns
        
        print df.dtypes
        
        print df['pHip']
        
        df.to_excel(strength_writer,cond)
        
    strength_writer.save()
    
    
def gather_strength_neg_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_net import get_strength_neg_values_from_info_nodes_file
    
    node_labels = np.array([line.strip() for line in open(ROI_coords_labels_file)],dtype = 'str')
    
    print node_labels
        
    strength_neg_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'strength_neg_values_by_cond.xls')
    
    strength_neg_writer = pd.ExcelWriter(strength_neg_filename)
    
    for cond in epi_cond:
    
        print cond
    
        strength_neg_cond_values = []
        
        for  subject_num in subject_nums:
            
            print subject_num
            
            node_prop_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"net_prop","Z_List-info_nodes.txt")
            
            print node_prop_file
            
            #0/0
            
            node_strength_neg_values = get_strength_neg_values_from_info_nodes_file(node_prop_file)
            
            print node_strength_neg_values
            
            
            strength_neg_cond_values.append(node_strength_neg_values)
            
            #print eff_val
            
            #eff_cond_values.append(eff_val)
            
        np_strength_neg_cond_values = np.array(strength_neg_cond_values,dtype = 'int64')
        
        print np_strength_neg_cond_values
        
        
        df = pd.DataFrame(np_strength_neg_cond_values,columns = node_labels,index = subject_nums, dtype = 'int64')
        
        print df
        
        print df.columns
        
        print df.dtypes
        
        print df['pHip']
        
        df.to_excel(strength_neg_writer,cond)
        
    strength_neg_writer.save()
    
def gather_strength_pos_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_net import get_strength_pos_values_from_info_nodes_file
    
    node_labels = np.array([line.strip() for line in open(ROI_coords_labels_file)],dtype = 'str')
    
    print node_labels
        
    strength_pos_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'strength_pos_values_by_cond.xls')
    
    strength_pos_writer = pd.ExcelWriter(strength_pos_filename)
    
    for cond in epi_cond:
    
        print cond
    
        strength_pos_cond_values = []
        
        for  subject_num in subject_nums:
            
            print subject_num
            
            node_prop_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"net_prop","Z_List-info_nodes.txt")
            
            print node_prop_file
            
            #0/0
            
            node_strength_pos_values = get_strength_pos_values_from_info_nodes_file(node_prop_file)
            
            print node_strength_pos_values
            
            
            strength_pos_cond_values.append(node_strength_pos_values)
            
            #print eff_val
            
            #eff_cond_values.append(eff_val)
            
        np_strength_pos_cond_values = np.array(strength_pos_cond_values,dtype = 'int64')
        
        print np_strength_pos_cond_values
        
        
        df = pd.DataFrame(np_strength_pos_cond_values,columns = node_labels,index = subject_nums, dtype = 'int64')
        
        print df
        
        print df.columns
        
        print df.dtypes
        
        print df['pHip']
        
        df.to_excel(strength_pos_writer,cond)
        
    strength_pos_writer.save()
    
def gather_degree_neg_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_net import get_degree_neg_values_from_info_nodes_file
    
    node_labels = np.array([line.strip() for line in open(ROI_coords_labels_file)],dtype = 'str')
    
    print node_labels
        
    degree_neg_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'degree_neg_values_by_cond2.xls')
    
    degree_neg_writer = pd.ExcelWriter(degree_neg_filename)
    
    for cond in epi_cond:
    
        print cond
    
        degree_neg_cond_values = []
        
        for  subject_num in subject_nums:
            
            print subject_num
            
            node_prop_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"net_prop","Z_List-info_nodes.txt")
            
            print node_prop_file
            
            #0/0
            
            node_degree_neg_values = get_degree_neg_values_from_info_nodes_file(node_prop_file)
            
            print node_degree_neg_values
            
            
            degree_neg_cond_values.append(node_degree_neg_values)
            
            #print eff_val
            
            #eff_cond_values.append(eff_val)
            
        np_degree_neg_cond_values = np.array(degree_neg_cond_values,dtype = 'int64')
        
        print np_degree_neg_cond_values
        
        
        df = pd.DataFrame(np_degree_neg_cond_values,columns = node_labels,index = subject_nums, dtype = 'int64')
        
        print df
        
        print df.columns
        
        print df.dtypes
        
        df.to_excel(degree_neg_writer,cond)
        
    degree_neg_writer.save()
    
def gather_degree_pos_values():

    import numpy as np
    import pandas as pd
    
    from dmgraphanalysis_nodes.utils_net import get_degree_pos_values_from_info_nodes_file
    
    node_labels = np.array([line.strip() for line in open(ROI_coords_labels_file)],dtype = 'str')
    
    print node_labels
        
    degree_pos_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'degree_pos_values_by_cond2.xls')
    
    degree_pos_writer = pd.ExcelWriter(degree_pos_filename)
    
    for cond in epi_cond:
    
        print cond
    
        degree_pos_cond_values = []
        
        for  subject_num in subject_nums:
            
            print subject_num
            
            node_prop_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_" + cond +"_subject_num_" + subject_num,"net_prop","Z_List-info_nodes.txt")
            
            print node_prop_file
            
            #0/0
            
            node_degree_pos_values = get_degree_pos_values_from_info_nodes_file(node_prop_file)
            
            print node_degree_pos_values
            
            
            degree_pos_cond_values.append(node_degree_pos_values)
            
            #print eff_val
            
            #eff_cond_values.append(eff_val)
            
        np_degree_pos_cond_values = np.array(degree_pos_cond_values,dtype = 'int64')
        
        print np_degree_pos_cond_values
        
        
        df = pd.DataFrame(np_degree_pos_cond_values,columns = node_labels,index = subject_nums, dtype = 'int64')
        
        print df
        
        print df.columns
        
        print df.dtypes
        
        df.to_excel(degree_pos_writer,cond)
        
    degree_pos_writer.save()
    
def compute_similarity_between_cond(simil_method = 'nmi'):

    import pandas as pd
    from igraph import compare_communities,Clustering
    
    
    from dmgraphanalysis_nodes.utils_net import read_lol_file

    simil_WWW_values = []
    
    simil_What_values = []
    
    simil_odor_values = []
    
    simil_recall_values = []
    
    for  subject_num in subject_nums:
         
         ### reading community
         odor_WWW_lol_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_Odor_Hit-WWW_subject_num_" + subject_num,"community_rada","Z_List.lol")
         
         odor_WWW_community_vect = read_lol_file(odor_WWW_lol_file)
         
         print odor_WWW_community_vect
         
         odor_What_lol_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_Odor_Hit-What_subject_num_" + subject_num,"community_rada","Z_List.lol")
         
         odor_What_community_vect = read_lol_file(odor_What_lol_file)
         
         print odor_What_community_vect
         
         recall_WWW_lol_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_Recall_Hit-WWW_subject_num_" + subject_num,"community_rada","Z_List.lol")
         
         recall_WWW_community_vect = read_lol_file(recall_WWW_lol_file)
         
         print recall_WWW_community_vect
         
         recall_What_lol_file = os.path.join(nipype_analyses_path,graph_analysis_name,"_cond_Recall_Hit-What_subject_num_" + subject_num,"community_rada","Z_List.lol")
         
         recall_What_community_vect = read_lol_file(recall_What_lol_file)
         
         print recall_What_community_vect
         
         ### compute simil
         
         if odor_WWW_community_vect.shape[0] == recall_WWW_community_vect.shape[0]:
         
            simil_WWW = compare_communities(Clustering(odor_WWW_community_vect),Clustering(recall_WWW_community_vect),method = simil_method)
         
            print simil_WWW
         
            simil_WWW_values.append(simil_WWW)
         
         else:
             
             print "Warning, community vect for %s WWW have different length"%subject_num
             
             sys.exit()
             
         if odor_What_community_vect.shape[0] == recall_What_community_vect.shape[0]:
         
            simil_What = compare_communities(Clustering(odor_What_community_vect),Clustering(recall_What_community_vect),method = simil_method)
         
            print simil_What
         
            simil_What_values.append(simil_What)
         
         else:
             
             print "Warning, community vect for %s What have different length"%subject_num
             
             sys.exit()
             
         if odor_WWW_community_vect.shape[0] == odor_What_community_vect.shape[0]:
         
            simil_odor = compare_communities(Clustering(odor_WWW_community_vect),Clustering(odor_What_community_vect),method = simil_method)
            
            print simil_odor
            
            simil_odor_values.append(simil_odor)
            
         else:
             
             print "Warning, community vect for %s odor have different length"%subject_num
             
             sys.exit()
           
           
         if recall_WWW_community_vect.shape[0] == recall_What_community_vect.shape[0]:
         
            simil_recall = compare_communities(Clustering(recall_WWW_community_vect),Clustering(recall_What_community_vect),method = simil_method)
            
            print simil_recall
            
            simil_recall_values.append(simil_recall)
            
         else:
             
             print "Warning, community vect for %s recall have different length"%subject_num
             
             sys.exit()
           
           
    #print simil_WWW_values
    
    #print simil_odor_values
    
    np_simil_values = np.vstack((np.array(simil_WWW_values,dtype = 'f'),np.array(simil_What_values,dtype = 'f'),np.array(simil_odor_values,dtype = 'f'),np.array(simil_recall_values,dtype = 'f')))
    
    print np_simil_values.shape
    
    df = pd.DataFrame(np.transpose(np_simil_values),columns = ["Simil_Odor-WWW_Recall-WWW","Simil_Odor-What_Recall-What","Simil_Odor-WWW_Odor-What","Simil_Recall-WWW_Recall-What"],index = subject_nums)
    
    df_filename = os.path.join(nipype_analyses_path,graph_analysis_name,'simil_'+ simil_method + '_values_by_cond.txt')
    
    df.to_csv(df_filename)
         
if __name__ =='__main__':
    
    
    ##### compute modular decomposition
    #print split_graph_analysis_name
        
    #main_workflow = create_wei_sig_modularity_workflow()
    

    ##main_workflow.write_graph(analysis_name +'graph.dot',graph2use='flat', format = 'svg')    
    #main_workflow.config['execution'] = {'remove_unnecessary_outputs':'false'}
    
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 1})
    ##main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 6})
    #main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})

    #### gathering results
    #gather_modularity_values()
    
    #gather_strength_values()
    #gather_strength_pos_values()
    #gather_strength_neg_values()
    
    gather_degree_pos_values()
    gather_degree_neg_values()
    
    #compute_similarity_between_cond(simil_method = "rand")
    
    