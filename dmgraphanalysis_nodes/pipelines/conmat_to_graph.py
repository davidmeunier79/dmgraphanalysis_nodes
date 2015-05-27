# -*- coding: utf-8 -*-
"""
Support function for net handling
"""
import sys
import time

import numpy as np
import scipy.sparse as sp 

import nipype.pipeline.engine as pe
    
from dmgraphanalysis_nodes.nodes.modularity import ComputeNetList,PrepRada
from dmgraphanalysis_nodes.nodes.modularity import CommRada,PlotIGraphModules
from dmgraphanalysis_nodes.nodes.modularity import NetPropRada
 
 
def create_pipeline_conmat_to_graph_density(correl_analysis_name,main_path,radatools_path,con_den = 1.0,multi = False,mod = True):

    pipeline = pe.Workflow(name=correl_analysis_name)
    pipeline.base_dir = main_path
    
    if multi == False:
        
        ################################################ density-based graphs
        
        #### net_list
        compute_net_List_den = pe.Node(interface = ComputeNetList(),name='compute_net_List_den')
        compute_net_List_den.inputs.density = con_den
        
        #pipeline.connect(convert_mat, 'conmat_file',compute_net_List_den, 'Z_cor_mat_file')
        
        ##### radatools ################################################################

        ### prepare net_list for radatools processing  
        prep_rada_den = pe.Node(interface = PrepRada(),name='prep_rada_den',iterfield = ["net_List_file"])
        prep_rada_den.inputs.radatools_path = radatools_path
        
        pipeline.connect(compute_net_List_den, 'net_List_file', prep_rada_den, 'net_List_file')
        
        
        if 'mod'==True:
                
            ### compute community with radatools
            community_rada_den = pe.Node(interface = CommRada(), name='community_rada_den',iterfield = ["Pajek_net_file"])
            #community_rada_den.inputs.optim_seq = radatools_optim
            community_rada_den.inputs.radatools_path = radatools_path
            
            pipeline.connect( prep_rada_den, 'Pajek_net_file',community_rada_den,'Pajek_net_file')
            
            #### plot_igraph_modules_rada
            plot_igraph_modules_rada_den = pe.Node(interface = PlotIGraphModules(),name='plot_igraph_modules_rada_den',iterfield = ['Pajek_net_file','rada_lol_file'])
            
            pipeline.connect(prep_rada_den, 'Pajek_net_file',plot_igraph_modules_rada_den,'Pajek_net_file')
            pipeline.connect(community_rada_den, 'rada_lol_file',plot_igraph_modules_rada_den,'rada_lol_file')
            
            #pipeline.connect(preproc, 'channel_coords_file',plot_igraph_modules_rada_den,'coords_file')
            #pipeline.connect(preproc, 'channel_names_file',plot_igraph_modules_rada_den,'labels_file')
            
        ############ compute network properties with rada
        net_prop_den = pe.Node(interface = NetPropRada(optim_seq = "A"), name = 'net_prop_den')
        net_prop_den.inputs.radatools_path = radatools_path
        
        pipeline.connect(prep_rada_den, 'Pajek_net_file',net_prop_den,'Pajek_net_file')
        
    else:
        
                
        ################################################ density-based graphs
        
        #### net_list
        compute_net_List_den = pe.MapNode(interface = ComputeNetList(),name='compute_net_List_den',iterfield = ["Z_cor_mat_file"])
        compute_net_List_den.inputs.density = con_den
        
        #pipeline.connect(convert_mat, 'conmat_file',compute_net_List_den, 'Z_cor_mat_file')
        
        ##### radatools ################################################################

        ### prepare net_list for radatools processing  
        prep_rada_den = pe.MapNode(interface = PrepRada(),name='prep_rada_den',iterfield = ["net_List_file"])
        prep_rada_den.inputs.radatools_path = radatools_path
        
        pipeline.connect(compute_net_List_den, 'net_List_file', prep_rada_den, 'net_List_file')
        
        
        if 'mod' in correl_analysis_name.split('_'):
                
            ### compute community with radatools
            community_rada_den = pe.MapNode(interface = CommRada(), name='community_rada_den',iterfield = ["Pajek_net_file"])
            community_rada_den.inputs.optim_seq = radatools_optim
            community_rada_den.inputs.radatools_path = radatools_path
            
            pipeline.connect( prep_rada_den, 'Pajek_net_file',community_rada_den,'Pajek_net_file')
            
            #### plot_igraph_modules_rada
            plot_igraph_modules_rada_den = pe.Node(interface = PlotIGraphModules(),name='plot_igraph_modules_rada_den',iterfield = ['Pajek_net_file','rada_lol_file'])
            
            pipeline.connect(prep_rada_den, 'Pajek_net_file',plot_igraph_modules_rada_den,'Pajek_net_file')
            pipeline.connect(community_rada_den, 'rada_lol_file',plot_igraph_modules_rada_den,'rada_lol_file')
            
            #pipeline.connect(preproc, 'channel_coords_file',plot_igraph_modules_rada_den,'coords_file')
            #pipeline.connect(preproc, 'channel_names_file',plot_igraph_modules_rada_den,'labels_file')
            
        ############ compute network properties with rada
        net_prop_den = pe.MapNode(interface = NetPropRada(optim_seq = "A"), name = 'net_prop_den')
        net_prop_den.inputs.radatools_path = radatools_path
        
        pipeline.connect(prep_rada_den, 'Pajek_net_file',net_prop_den,'Pajek_net_file')


    return pipeline

    
