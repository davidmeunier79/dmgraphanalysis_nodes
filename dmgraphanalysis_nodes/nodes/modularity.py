# -*- coding: utf-8 -*-

#from dmgraphanalysis.plot_igraph import *

import rpy,os
import nibabel as nib
import numpy as np


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec

#from nipype.interfaces.base import CommandLine, CommandLineInputSpec ### not used but should be done for PrepRada

from nipype.interfaces.base import traits, File, TraitedSpec, isdefined
    
from nipype.utils.filemanip import split_filename as split_f
    
######################################################################################## ComputeNetList ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import return_net_list
from dmgraphanalysis_nodes.utils_net import export_List_net_from_list,export_Louvain_net_from_list

class ComputeNetListInputSpec(BaseInterfaceInputSpec):
    
    Z_cor_mat_file = File(exists=True, desc='Normalized correlation matrix', mandatory=True)
    
    coords_file = File(exists=True, desc='Corresponding coordiantes', mandatory=True)
    
class ComputeNetListOutputSpec(TraitedSpec):
    
    net_List_file = File(exists=True, desc="net list for radatools")
    
    net_Louvain_file = File(exists=True, desc="net list for Louvain")
    
class ComputeNetList(BaseInterface):
    
    """
    Format correlation matrix to a list format i j weight (integer)
    """

    input_spec = ComputeNetListInputSpec
    output_spec = ComputeNetListOutputSpec

    def _run_interface(self, runtime):
                
        Z_cor_mat_file = self.inputs.Z_cor_mat_file
        coords_file = self.inputs.coords_file
        
        print "loading Z_cor_mat_file"
        
        Z_cor_mat = np.load(Z_cor_mat_file)
        
        print 'load coords'
        
        coords = np.array(np.loadtxt(coords_file),dtype = int)
        
        ## compute Z_list 
        
        print "computing Z_list by thresholding Z_cor_mat"
        
        Z_list = return_net_list(Z_cor_mat)
        
        ## Z correl_mat as list of edges
        
        print "saving Z_list as list of edges"
        
        net_List_file = os.path.abspath('Z_List.txt')
        
        export_List_net_from_list(net_List_file,Z_list)
        
        ### Z correl_mat as Louvain format
        
        print "saving Z_list as Louvain format"
        
        net_Louvain_file = os.path.abspath('Z_Louvain.txt')
        
        export_Louvain_net_from_list(net_Louvain_file,Z_list,coords)
        
        return runtime
        
        #return mean_masked_ts_file,subj_coord_rois_file
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["net_List_file"] = os.path.abspath("Z_List.txt")
        outputs["net_Louvain_file"] = os.path.abspath("Z_Louvain.txt")
    
        return outputs
    
######################################################################################## ComputeIntNetList ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import return_int_net_list
from dmgraphanalysis_nodes.utils_net import export_List_net_from_list,export_Louvain_net_from_list

class ComputeIntNetListInputSpec(BaseInterfaceInputSpec):
    
    int_mat_file = File(exists=True, desc='Integer matrix', mandatory=True)
    
    #coords_file = File(exists=True, desc='Corresponding coordiantes', mandatory=True)
    
    threshold = traits.Int(exists = True, desc = "Interger Value (optional) for thresholding", mandatory = False) 
    
class ComputeIntNetListOutputSpec(TraitedSpec):
    
    net_List_file = File(exists=True, desc="net list for radatools")
    
    #net_Louvain_file = File(exists=True, desc="net list for Louvain")
    
class ComputeIntNetList(BaseInterface):
    
    """
    Format integer matrix to a list format i j weight 
    Option for thresholding 
    """

    input_spec = ComputeIntNetListInputSpec
    output_spec = ComputeIntNetListOutputSpec

    def _run_interface(self, runtime):
                
        int_mat_file = self.inputs.int_mat_file
        #coords_file = self.inputs.coords_file
        threshold = self.inputs.threshold
        
        
        print "loading int_mat_file"
        
        int_mat = np.load(int_mat_file)
        
        #print 'load coords'
        
        #coords = np.array(np.loadtxt(coords_file),dtype = int)
        
        ## compute int_list 
        
        if not isdefined(threshold):
            
            threshold = 0
        
        
        int_list = return_int_net_list(int_mat,threshold)
            
        ## int correl_mat as list of edges
        
        print "saving int_list as list of edges"
        
        net_List_file = os.path.abspath('int_List.txt')
        
        export_List_net_from_list(net_List_file,int_list)
        
        #### int correl_mat as Louvain format
        #print "saving int_list as Louvain format"
        
        #net_Louvain_file = os.path.abspath('int_Louvain.txt')
        
        #export_Louvain_net_from_list(net_Louvain_file,int_list,coords)
        
        return runtime
        
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["net_List_file"] = os.path.abspath("int_List.txt")
        #outputs["net_Louvain_file"] = os.path.abspath("int_Louvain.txt")
    
        return outputs
    
######################################################################################## PrepRada ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import return_net_list
from dmgraphanalysis_nodes.utils_net import export_List_net_from_list,export_Louvain_net_from_list


class PrepRadaInputSpec(BaseInterfaceInputSpec):
    
    radatools_path = traits.String(exists=True,desc='path to radatools software - installed externally ', mandatory=True, position = 0, argstr="%s")
    
    net_List_file = File(exists=True, desc='List of edges describing net in format i j weight', mandatory=True)
    
class PrepRadaOutputSpec(TraitedSpec):
    
    Pajek_net_file = File(exists=True, desc="net description in Pajek format, generated by radatools")
    
class PrepRada(BaseInterface):
    
    """
    Format net list (format i j weight) to Pajek file
    """
    input_spec = PrepRadaInputSpec
    output_spec = PrepRadaOutputSpec

    
    #def __init__(self):
    
        #self._cmd = os.path.join(radatools_path,'01-Prepare_Network','List_To_Net.exe')
        
    def _run_interface(self, runtime):
                
        radatools_path = self.inputs.radatools_path
        net_List_file = self.inputs.net_List_file
        
        
        path, fname, ext = split_f(net_List_file)
    
        Pajek_net_file = os.path.abspath(fname + '.net')
    
        cmd = os.path.join(radatools_path,'01-Prepare_Network','List_To_Net.exe') + ' ' + net_List_file + ' ' + Pajek_net_file + ' U'
        
        print "defining command " + cmd
    
        os.system(cmd)
    
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        path, fname, ext = split_f(self.inputs.net_List_file)
    
        outputs["Pajek_net_file"] = os.path.abspath(fname + '.net')
    
        return outputs

        
        ########### do not work, how to specify the path of the executables???? ######
        
#class PrepRadaInputSpec(CommandLineInputSpec):
    
    #radatools_path = traits.String(exists=True,desc='path to radatools software - installed externally ', mandatory=True, position = 0, argstr="%s")
    
    #net_List_file = File(exists=True, desc='List of edges describing net in format i j weight', mandatory=True)
    
#class PrepRadaOutputSpec(TraitedSpec):
    
    #Pajek_net_file = File(exists=True, desc="net description in Pajek format, generated by radatools")
    
#class PrepRada(CommandLine):
    
    #"""
    #Format net list (format i j weight) to Pajek file
    #"""
    
    #_cmd = os.path.join('01-Prepare_Network','List_To_Net.exe')
    
    #input_spec = PrepRadaInputSpec
    #output_spec = PrepRadaOutputSpec

    #def _run_interface(self, runtime):
                
        #radatools_path = self.inputs.radatools_path
        #net_List_file = self.inputs.net_List_file
        
        #path, fname, ext = split_f(net_List_file)
    
        #Pajek_net_file = os.path.abspath(fname + '.net')
    
        #self.args = [net_List_file , Pajek_net_file ,' U' ]
        
        #self._cmdpath = radatools_path
    
        #return runtime
        
    #def _list_outputs(self):
        
        #outputs = self._outputs().get()
        
        #net_List_file = self.inputs.net_List_file
        
        #path, fname, ext = split_f(net_List_file)
    
        #outputs["Pajek_net_file"] = os.path.abspath(fname + '.net')
    
        #return outputs

######################################################################################## CommRada ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import return_net_list
from dmgraphanalysis_nodes.utils_net import export_List_net_from_list,export_Louvain_net_from_list


class CommRadaInputSpec(BaseInterfaceInputSpec):
    
    radatools_path = traits.String(exists=True,desc='path to radatools software - installed externally ', mandatory=True, position = 0, argstr="%s")
    
    Pajek_net_file = File(exists=True, desc='net description in Pajek format', mandatory=True)
    
    optim_seq = traits.String(exists=True, desc = "Optimisation sequence, see radatools documentation for more information")
    
class CommRadaOutputSpec(TraitedSpec):
    
    rada_lol_file = File(exists=True, desc="modularity structure description, generated by radatools")
    rada_log_file = File(exists=True, desc="optimisation steps, generated by radatools")
    
class CommRada(BaseInterface):
    
    """
    Launch community detection on Pajek file with given optimisation args
    """
    input_spec = CommRadaInputSpec
    output_spec = CommRadaOutputSpec

    
    #def __init__(self):
    
        #self._cmd = os.path.join(radatools_path,'01-Prepare_Network','List_To_Net.exe')
        
    def _run_interface(self, runtime):
                
        radatools_path = self.inputs.radatools_path
        Pajek_net_file = self.inputs.Pajek_net_file
        
        optim_seq = self.inputs.optim_seq
        
        path, fname, ext = split_f(Pajek_net_file)
    
        rada_lol_file = os.path.abspath(fname + '.lol')
        rada_log_file = os.path.abspath(fname + '.log')
        
        cmd = os.path.join(radatools_path,'02-Find_Communities','Communities_Detection.exe')  + ' v ' + optim_seq + ' ' + Pajek_net_file + ' ' + rada_lol_file + ' > ' + rada_log_file
        
        print "defining command " + cmd
    
        os.system(cmd)
    
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        path, fname, ext = split_f(self.inputs.Pajek_net_file)
    
        rada_lol_file = os.path.abspath(fname + '.lol')
        rada_log_file = os.path.abspath(fname + '.log')
        
        outputs["rada_lol_file"] = os.path.abspath(fname + '.lol')
        outputs["rada_log_file"] = os.path.abspath(fname + '.log')
    
        return outputs

######################################################################################## NetPropRada ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import return_net_list
from dmgraphanalysis_nodes.utils_net import export_List_net_from_list,export_Louvain_net_from_list


class NetPropRadaInputSpec(BaseInterfaceInputSpec):
    
    radatools_path = traits.String(exists=True,desc='path to radatools software - installed externally ', mandatory=True, position = 0, argstr="%s")
    
    Pajek_net_file = File(exists=True, desc='net description in Pajek format', mandatory=True)
    
    optim_seq = traits.String('all 2',usedefault = True, exists=True, desc = "Optimisation sequence, see radatools documentation for more information")
    
class NetPropRadaOutputSpec(TraitedSpec):
    
    rada_log_file = File(exists=True, desc="network properties log, generated by radatools")
    
class NetPropRada(BaseInterface):
    
    """
    Launch Network properties on Pajek file with given parameters (see Network_Properties in Radatools)
    """
    input_spec = NetPropRadaInputSpec
    output_spec = NetPropRadaOutputSpec

    
    #def __init__(self):
    
        #self._cmd = os.path.join(radatools_path,'01-Prepare_Network','List_To_Net.exe')
        
    def _run_interface(self, runtime):
                
        radatools_path = self.inputs.radatools_path
        Pajek_net_file = self.inputs.Pajek_net_file
        
        optim_seq = self.inputs.optim_seq
        
        path, fname, ext = split_f(Pajek_net_file)
    
        rada_log_file = os.path.abspath(fname + '.log')
        
        cmd = os.path.join(radatools_path,'04-Other_Tools','Network_Properties.exe') + ' ' + Pajek_net_file + ' ' + optim_seq + ' > ' + rada_log_file
        
        #+ ' > '+ rada_res_file
        
        print "defining command " + cmd
    
        os.system(cmd)
    
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        path, fname, ext = split_f(self.inputs.Pajek_net_file)
    
        outputs["rada_log_file"] = os.path.abspath(fname + '.log')
    
        return outputs

######################################################################################## ComputeNodeRoles ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix
from dmgraphanalysis_nodes.utils_mod import compute_roles

class ComputeNodeRolesInputSpec(BaseInterfaceInputSpec):
    
    rada_lol_file = traits.String(exists=True,desc='lol file, describing modular structure of the network', mandatory=True, position = 0, argstr="%s")
    Pajek_net_file = File(exists=True, desc='net description in Pajek format', mandatory=True)
    
    role_type =  traits.Enum('Amaral_roles', '4roles', 
    desc='definition of node roles, Amaral_roles = original 7 roles defined for transport network (useful for big network), 4_roles defines only provincial/connecteur from participation coeff',
    usedefault=True)
    
class ComputeNodeRolesOutputSpec(TraitedSpec):
    
    node_roles_file = File(exists=True, desc="node roles with an integer code")
    all_Z_com_degree_file = File(exists=True, desc="value of quantity, describing the hub/non-hub role of the nodes")
    all_participation_coeff_file  = File(exists=True, desc="value of quality, descibing the provincial/connector role of the nodes")
    
class ComputeNodeRoles(BaseInterface):
    
    """
    compute node roles from lol modular partition and original network
    """
    input_spec = ComputeNodeRolesInputSpec
    output_spec = ComputeNodeRolesOutputSpec

    def _run_interface(self, runtime):
                
        rada_lol_file = self.inputs.rada_lol_file
        Pajek_net_file = self.inputs.Pajek_net_file
        
        
        print 'Loading Pajek_net_file for reading node_corres'
        
        node_corres,sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
        
        print sparse_mat.todense()
        
        print node_corres.shape,sparse_mat.todense().shape
        
        print "Loading community belonging file " + rada_lol_file

        community_vect = read_lol_file(rada_lol_file)
        
        print community_vect
        
        print "Computing node roles"
        
        node_roles,all_Z_com_degree,all_participation_coeff = compute_roles(community_vect,sparse_mat, role_type = self.inputs.role_type)
        
        print node_roles
        
        
        node_roles_file = os.path.abspath('node_roles.txt')
        
        np.savetxt(node_roles_file,node_roles,fmt = '%d')
        
        
        all_Z_com_degree_file = os.path.abspath('all_Z_com_degree.txt')
        
        np.savetxt(all_Z_com_degree_file,all_Z_com_degree,fmt = '%f')
        
        
        all_participation_coeff_file = os.path.abspath('all_participation_coeff.txt')
        
        np.savetxt(all_participation_coeff_file,all_participation_coeff,fmt = '%f')
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["node_roles_file"] = os.path.abspath('node_roles.txt')
        outputs["all_Z_com_degree_file"] = os.path.abspath('all_Z_com_degree.txt')
        outputs["all_participation_coeff_file"] = os.path.abspath('all_participation_coeff.txt')
        
        return outputs

######################################################################################## PlotIGraphModules ##################################################################################################################

from dmgraphanalysis_nodes.utils_net import read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix
from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_all_modules,plot_3D_igraph_single_modules

import csv

class PlotIGraphModulesInputSpec(BaseInterfaceInputSpec):
    
    rada_lol_file = File(exists=True, desc="modularity structure description, generated by radatools", mandatory=True)
    Pajek_net_file = File(exists=True, desc='net description in Pajek format', mandatory=True)
    coords_file = File(exists=True, desc="txt file containing the coords of the nodes", mandatory=False)
    labels_file = File(exists=True, desc="txt file containing the labels of the nodes (full description)", mandatory=False)
    node_roles_file = File(exists=True, desc="txt file containing the roles of the nodes (integer labels)", mandatory=False)
    
class PlotIGraphModulesOutputSpec(TraitedSpec):
    
    #Z_list_single_modules_files = traits.List(File(exists=True), desc="graphical representation in space of each module independantly")    
    Z_list_all_modules_files = traits.List(File(exists=True), desc="graphical representation in space from different point of view of all modules together")
    
class PlotIGraphModules(BaseInterface):
    
    """
    Graphical representation of the modular structure of the network. 
    If coords are provided, plot in space, otherwise use topological space.
    """
    
    #If node labels are provided, add them to the graph
    
    input_spec = PlotIGraphModulesInputSpec
    output_spec = PlotIGraphModulesOutputSpec

    
    def _run_interface(self, runtime):
                
                
        rada_lol_file = self.inputs.rada_lol_file
        Pajek_net_file = self.inputs.Pajek_net_file
        coords_file = self.inputs.coords_file
        labels_file = self.inputs.labels_file
        node_roles_file = self.inputs.node_roles_file
        
        print 'Loading node_corres and Z list'
        
        node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
        
        print np.min(node_corres),np.max(node_corres)
        
        print node_corres.shape
        print node_corres
        
        print Z_list
        print Z_list.shape 
        
        print 'Loading coords'
        
        #print 'Loading gm mask coords'
        
        #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
        
        #print gm_mask_coords.shape
        
        print "Loading community belonging file" + rada_lol_file

        community_vect = read_lol_file(rada_lol_file)
        
        print community_vect
        print community_vect.shape
        print np.min(community_vect),np.max(community_vect)
        
        if isdefined(coords_file):
            
            print 'extracting node coords'
            
            #with open(coords_file, 'Ur') as f:
                #coords_list = list(tuple(map(int,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
            
            
            
            coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
            print coords.shape
            
            node_coords = coords[node_corres,:]
            print node_coords.shape
            
        else :
            node_coords = np.array([])
            
            
        if isdefined(labels_file):
            
            print 'extracting node labels'
                   
            labels = [line.strip() for line in open(labels_file)]
            print labels
            
            node_labels = np.array(labels,dtype = 'string')[node_corres].tolist()
            print len(node_labels)
            
        else :
            node_labels = []
            
        if isdefined(node_roles_file):
        
            print 'extracting node roles'
                    
            node_roles = np.array(np.loadtxt(node_roles_file),dtype = 'int64')
            
            #print node_roles 
            
        else:
        
            node_roles = np.array([])
           
            
        #print node_roles 
        
        print "plotting conf_cor_mat_modules_file with igraph"
        
        #Z_list_single_modules_files = plot_3D_igraph_single_modules(community_vect,Z_list,node_coords,node_labels,nb_min_nodes_by_module = 3)
        Z_list_all_modules_files = plot_3D_igraph_all_modules(community_vect,Z_list,node_coords,node_labels,node_roles = node_roles)
        Z_list_all_modules_files = plot_3D_igraph_all_modules(community_vect,Z_list,node_labels= node_labels,node_roles = node_roles, layout = 'FR')
        
        #self.Z_list_single_modules_files = Z_list_single_modules_files
        self.Z_list_all_modules_files = Z_list_all_modules_files
                
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        #outputs["Z_list_single_modules_files"] = self.Z_list_single_modules_files
        outputs["Z_list_all_modules_files"] =self.Z_list_all_modules_files 
        
        return outputs

################################################# Louvain method  ###########################################

#def convert_list_Louvain(Louvain_list_file,louvain_bin_path):
    #import os
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    ##path, fname, ext = '','',''
    #path, fname, ext = split_f(Louvain_list_file)
    
    #Louvain_bin_file = os.path.abspath(fname + '.bin')
    
    #Louvain_node_file = os.path.abspath(fname  + '.node.txt')
    
    #Louvain_conf_file = os.path.abspath(fname  + '.conf')
    
    #cmd = os.path.join(louvain_bin_path,'slicer') + ' -i ' + Louvain_list_file + ' -o ' +  Louvain_bin_file + ' -n ' + Louvain_node_file + ' -c ' + Louvain_conf_file + ' -u' 
    
    #print "executing command " + cmd
    
    #os.system(cmd)
    
    #return Louvain_bin_file,Louvain_node_file,Louvain_conf_file
    
#def community_list_Louvain(Louvain_bin_file,Louvain_conf_file,louvain_bin_path):
    
    #import os
    #from nipype.utils.filemanip import split_filename as split_f
    
    ##path, fname, ext = '','',''
    #path, fname, ext = split_f(Louvain_bin_file)
    
    #Louvain_mod_file = os.path.abspath(fname + '.mod')
    
    #cmd = os.path.join(louvain_bin_path,'community') + ' ' + Louvain_bin_file + ' ' +  Louvain_conf_file + ' > ' + Louvain_mod_file
    
    #print "executing command " + cmd
    
    #os.system(cmd)
    
    #return Louvain_mod_file
    
    
#def export_mod_mask_file(Louvain_mod_file,Louvain_node_file,coords_file,mask_file):

    #import numpy as np
    #import nibabel as nib
    #import os
    
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_Louvain_corres_nodes,read_mod_file

    #print 'Loading node_corres'
    
    #node_corres = read_Louvain_corres_nodes(Louvain_node_file)
    
    #print node_corres
    #print node_corres.shape
    
    #print 'Loading coords'
    
    #coords = np.array(np.loadtxt(coords_file),dtype = int)
    
    #print coords.shape
    
    #print 'Loading mask parameters'
    
    #mask = nib.load(mask_file)
    
    #data_mask_shape = mask.get_data().shape
    
    #mask_header = mask.get_header().copy()
    
    #mask_affine = np.copy(mask.get_affine())
    
    #print "Loading community belonging file" + Louvain_mod_file

    #community_vect = read_mod_file(Louvain_mod_file)
    
    ##print community_vect
    #print community_vect.shape
    
    #print "transforming to nii file"
    #mod_mask_data = return_mod_mask_corres(community_vect,node_corres,coords,data_mask_shape)

    ##print mod_mask_data
    #print mod_mask_data.shape


    #print "saving npy file"

    #mod_mask_file = os.path.abspath("mod_mask_data.npy")

    #np.save(mod_mask_file ,np.array(mod_mask_data,dtype = int))
    
    #print "saving nii file"

    #mod_mask_file = os.path.abspath("mod_mask_data.nii")

    #nib.save(nib.Nifti1Image(np.array(mod_mask_data,dtype = int),mask_affine,mask_header),mod_mask_file)

    #print "returning"

    #return mod_mask_file
    

################################## radatools  ########################################################

    
    
#def community_radatools(Pajek_net_file,optim_seq,radatools_comm_path):
    
    #import os
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    ##path, fname, ext = '','',''
    #path, fname, ext = split_f(Pajek_net_file)
    
    #rada_lol_file = os.path.abspath(fname + '.lol')
    #rada_log_file = os.path.abspath(fname + '.log')
    
    ##cmd = os.path.join(radatools_comm_path,'Communities_Detection.exe') + ' v WS f 1 ' + Pajek_net_file + ' ' + rada_lol_file + ' > ' + rada_log_file
    #cmd = os.path.join(radatools_comm_path,'Communities_Detection.exe') + ' v ' + optim_seq + ' ' + Pajek_net_file + ' ' + rada_lol_file + ' > ' + rada_log_file
    
    
    #print "executing command " + cmd
    
    #os.system(cmd)
    
    #return rada_lol_file,rada_log_file
    
    
#def export_lol_mask_file(rada_lol_file,Pajek_net_file,coords_file,mask_file):

    #import numpy as np
    
    #import nibabel as nib
    #import os
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes

    #print 'Loading Pajek_net_file for reading node_corres'
    
    #node_corres = read_Pajek_corres_nodes(Pajek_net_file)
    
    #print node_corres.shape
    
    #print 'Loading coords'
    
    #coords = np.array(np.loadtxt(coords_file),dtype = int)
    
    #print coords.shape
    
    #print 'Loading mask parameters'
    
    #mask = nib.load(mask_file)
    
    #data_mask_shape = mask.get_data().shape
    
    #mask_header = mask.get_header().copy()
    
    #mask_affine = np.copy(mask.get_affine())
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    #print community_vect
    
    #print "transforming to nii file"
    #lol_mask_data = return_mod_mask_corres(community_vect,node_corres,coords,data_mask_shape)

    ##print lol_mask_data
    #print lol_mask_data.shape


    ##print "saving npy file"

    ##mod_mask_file = os.path.abspath("mod_mask_data.npy")

    ##np.save(mod_mask_file ,np.array(mod_mask_data,dtype = int))
    
    #print "saving nii file"

    #lol_mask_file = os.path.abspath("lol_mask_data.nii")

    #nib.save(nib.Nifti1Image(np.array(lol_mask_data,dtype = int),mask_affine,mask_header),lol_mask_file)

    ##print "returning"

    ##lol_mask_file = ""
    
    #return lol_mask_file
    
#def export_lol_mask_file_coords_net(rada_lol_file,Pajek_net_file,coords_net_file,mask_file):

    #import numpy as np
    
    #import nibabel as nib
    #import os
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres_rel_coords,return_mod_mask_corres_rel_coords_neighbourhood,read_lol_file,read_Pajek_rel_coords,read_Pajek_corres_nodes

    ##print 'Loading Pajek_net_file for reading node_corres'
    
    ##node_corres = read_Pajek_corres_nodes(Pajek_net_file)
    
    ##print node_corres.shape
    
    
    #print 'Loading coords'
    
    #node_rel_coords = read_Pajek_rel_coords(coords_net_file)
    
    #print node_rel_coords
    
    #print 'Loading mask parameters'
    
    #mask = nib.load(mask_file)
    
    #data_mask_shape = mask.get_data().shape
    
    #mask_header = mask.get_header().copy()
    
    #mask_affine = np.copy(mask.get_affine())
    
    #print data_mask_shape
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    #print community_vect
    
    #print "transforming to nii file"
    ##lol_mask_data = return_mod_mask_corres_rel_coords(community_vect,node_rel_coords,data_mask_shape[:-1])
    #lol_mask_data = return_mod_mask_corres_rel_coords_neighbourhood(community_vect,node_rel_coords,data_mask_shape[:-1])

    ##print lol_mask_data
    #print lol_mask_data.shape


    ##print "saving npy file"

    ##mod_mask_file = os.path.abspath("mod_mask_data.npy")

    ##np.save(mod_mask_file ,np.array(mod_mask_data,dtype = int))
    
    #print "saving nii file"

    #lol_mask_file = os.path.abspath("lol_mask_data.nii")

    #nib.save(nib.Nifti1Image(np.array(lol_mask_data,dtype = int),mask_affine,mask_header),lol_mask_file)

    ##print "returning"

    ##lol_mask_file = ""
    
    #return lol_mask_file
    

#def get_modularity_value_from_lol_file(modularity_file):
    
    #with open(modularity_file,'r') as f:
        
        #for line in f.readlines():
        
            #split_line = line.strip().split(' ')
            
            #print split_line
            
            #if split_line[0] == 'Q':
            
                #print "Found modularity value line"
                
                #return split_line[2] 
                
                
        #print "Unable to find modularity line in file, returning -1"
        
        #return -1.0
############################## computation on modules 
    
    
#def compute_mod_average_ts_louvain(ts_mat_file,coords_file,Louvain_mod_file,Louvain_node_file):

    #import os
    #import numpy as np

    #from dmgraphanalysis.utils_cor import compute_average_ts_by_module_corres,read_mod_file,read_Louvain_corres_nodes

    #print 'load coords'
    
    #coords = np.loadtxt(coords_file)
    
    #print 'load time series'
    
    #ts_mat = np.load(ts_mat_file)
    
    #print 'load Louvain community'
    
    #community_vect = read_mod_file(Louvain_mod_file)
    
    #print 'load node corres Louvain node file'
    
    #node_corres = read_Louvain_corres_nodes(Louvain_node_file)
    
    #print 'compute_average_ts_by_module'
    
    #mod_average_ts,mod_average_coords = compute_average_ts_by_module_corres(ts_mat,coords,community_vect,node_corres)

    ##print mod_average_ts
    ##print mod_average_coords


    #print mod_average_ts.shape
    #print mod_average_coords.shape

    #print "saving mod average time series"
    #mod_average_ts_file = os.path.abspath('mod_average_ts_mat.npy')

    #np.save(mod_average_ts_file,mod_average_ts)


    #print "saving mod average coordinates"
    #mod_average_coords_file = os.path.abspath('mod_average_coords.txt')

    #np.savetxt(mod_average_coords_file,mod_average_coords)

    #return mod_average_ts_file,mod_average_coords_file

#def compute_mod_average_ts_rada(ts_mat_file,coords_file,rada_lol_file,Pajek_net_file):

    #import os
    #import numpy as np

    #from dmgraphanalysis.utils_cor import compute_average_ts_by_module_corres,read_lol_file,read_Pajek_corres_nodes

    #print 'load coords'
    
    #coords = np.loadtxt(coords_file)
    
    #print 'load time series'
    
    #ts_mat = np.load(ts_mat_file)
    
    #print 'load Louvain community'
    
    #community_vect = read_lol_file(rada_lol_file)
    
    #print 'load node corres from Pajek file'
    
    #node_corres = read_Pajek_corres_nodes(Pajek_net_file)
    
    
    #print 'compute_average_ts_by_module_corres'
    
    #mod_average_ts,mod_average_coords = compute_average_ts_by_module_corres(ts_mat,coords,community_vect,node_corres)

    ##print mod_average_ts
    ##print mod_average_coords


    #print mod_average_ts.shape
    #print mod_average_coords.shape

    #print "saving mod average time series"
    #mod_average_ts_file = os.path.abspath('mod_average_ts_mat.npy')

    #np.save(mod_average_ts_file,mod_average_ts)


    #print "saving mod average coordinates"
    #mod_average_coords_file = os.path.abspath('mod_average_coords.txt')

    #np.savetxt(mod_average_coords_file,mod_average_coords)

    
    #return mod_average_ts_file,mod_average_coords_file

#def compute_mod_cor_mat(mod_average_ts_file,regressor_file):

    #import os
    #import numpy as np

    #from dmgraphanalysis.utils_cor import compute_weighted_cor_mat_non_zeros

    #print 'load regressor_vect'
    
    #regressor_vect = np.loadtxt(regressor_file)
    
    #print 'load mod_average_ts_mat'
    
    #mod_average_ts = np.load(mod_average_ts_file)
    
    #print 'compute_weighted_cor_mat_non_zeros'
    
    #mod_cor_mat,mod_Z_cor_mat = compute_weighted_cor_mat_non_zeros(np.transpose(mod_average_ts),regressor_vect)

    #print mod_cor_mat
    #print mod_Z_cor_mat

    #print "saving mod cor mat"
    #mod_cor_mat_file = os.path.abspath('mod_cor_mat.txt')

    #np.savetxt(mod_cor_mat_file,mod_cor_mat,fmt = '%2.2f')

    #print "saving mod Z cor mat"
    #mod_Z_cor_mat_file = os.path.abspath('mod_Z_cor_mat.txt')

    #np.savetxt(mod_Z_cor_mat_file,mod_Z_cor_mat,fmt = '%2.2f')

    
    #return mod_cor_mat_file,mod_Z_cor_mat_file

    
################################################################### plotting ###############################################
#def plot_dist_matrix(dist_mat_file):

    #import os
    #import numpy as np
    
    #import nibabel as nib
    
    #from nipype.utils.filemanip import split_filename as split_f
    #from dmgraphanalysis.utils_plot import plot_cormat,plot_hist
    
    ########## dist_mat
    
    #dist_mat = np.load(dist_mat_file)
    
    ##### heatmap 
    
    #print 'plotting distance matrix heatmap'
    
    #plot_heatmap_dist_mat_file =  os.path.abspath('heatmap_distance_matrix.eps')
    
    #plot_cormat(plot_heatmap_dist_mat_file,dist_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting distance matrix histogram'
     
    #plot_hist_dist_mat_file = os.path.abspath('hist_distance_matrix.eps')
    
    #plot_hist(plot_hist_dist_mat_file,dist_mat,nb_bins = 100)
    
    #return plot_hist_dist_mat_file,plot_heatmap_dist_mat_file
    
        
############################ plot igraph #####################################""
    
#def plot_igraph_modules_conf_cor_mat_louvain(Louvain_mod_file,Louvain_node_file,coords_file,net_Louvain_file,gm_mask_coords_file):

    #"""
    #Special Louvain algo
    #Needs to be modified to match with plotting for Rada results (coords and gm_coords are redondnant, etc.)
    #"""
    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
    
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_Louvain_corres_nodes,read_mod_file,read_Louvain_net_file
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_modules_Z_list

    #print 'Loading node_corres'
    
    #node_corres = read_Louvain_corres_nodes(Louvain_node_file)
    
    #print node_corres
    #print node_corres.shape
    
    #print 'Loading coords'
    
    ##with open(coords_file, 'Ur') as f:
        ##coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = int)
    
    #print coords.shape
    
    #print 'Loading gm mask coords'
    
    #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    #print gm_mask_coords.shape
    
    #print "Loading community belonging file" + Louvain_mod_file

    #community_vect = read_mod_file(Louvain_mod_file)
    
    ##print community_vect
    #print community_vect.shape
    
    #print "loading net_Louvain_file as list"
    
    #Z_list = read_Louvain_net_file(net_Louvain_file)
    
    ##print Z_list
    
    #print 'extracting node coords'
    
    #node_coords = coords[node_corres,:]
    
    #print node_coords.shape
    
    #print "plotting conf_cor_mat_modules_file with igraph"
    
    #Z_list_all_modules_files = plot_3D_igraph_modules_Z_list(community_vect,node_coords,Z_list,gm_mask_coords)
    
    ##Z_list_all_modules_files = ''
    ##conf_cor_mat_big_modules_file = ''
    
    #return Z_list_all_modules_files
    
    
#def plot_igraph_modules_conf_cor_mat_rada(rada_lol_file,Pajek_net_file,coords_file):

    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_all_modules_coomatrix_rel_coords,plot_3D_igraph_single_modules_coomatrix_rel_coords

    #print 'Loading node_corres and Z list'
    
    #node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    #print np.min(node_corres),np.max(node_corres)
    
    #print node_corres.shape
    
    #print Z_list.shape 
    
    #print 'Loading coords'
    
    
    ##with open(coords_file, 'Ur') as f:
        ##coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    ##print 'Loading gm mask coords'
    
    ##gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    ##print gm_mask_coords.shape
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    #print community_vect.shape
    #print np.min(community_vect),np.max(community_vect)
    
    #print 'extracting node coords'
    
    #node_coords = coords[node_corres,:]
    
    #print node_coords.shape
    
    #print "plotting conf_cor_mat_modules_file with igraph"
    #Z_list_single_modules_files = plot_3D_igraph_single_modules_coomatrix_rel_coords(community_vect,node_coords,Z_list)
    #Z_list_all_modules_files = plot_3D_igraph_all_modules_coomatrix_rel_coords(community_vect,node_coords,Z_list)
    
    #return Z_list_single_modules_files,Z_list_all_modules_files
    
#def plot_igraph_modules_coclass_rada(rada_lol_file,Pajek_net_file,gm_mask_coords_file,labels_file):
    
    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix,read_List_net_file
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_single_modules_coomatrix_rel_coords,plot_3D_igraph_all_modules_coomatrix_rel_coords
##    from dmgraphanalysis.plot_igraph import plot_3D_igraph_modules_net_list

    #print 'loading labels'
    
    #labels = [line.strip() for line in open(labels_file)]
    
    
    #print 'Loading node_corres'
    
    #node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    
    ##print np.min(node_corres),np.max(node_corres)
    ##print node_corres.shape
    
    ##print Z_list.shape
    #print 'Loading gm mask coords'
    
    #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    ##print gm_mask_coords.shape
    
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    ##print community_vect.shape
    
    #print "loading net_list_net as list"
    
    ##net_list = read_List_net_file(net_list_file)
    
    ##print net_list
    
    #print 'extracting node coords'
    
    #node_coords = gm_mask_coords[node_corres,:]
    
    ##print node_coords.shape
    
    #np_labels = np.array(labels,dtype = 'string')
    
    #node_labels = np_labels[node_corres,:]
    
    ##print node_labels.shape
    
    ##print len(labels),node_labels.shape
    
    
    #print "plotting Z_cor_mat_modules_file with igraph"
    
    #coclass_single_modules_files = plot_3D_igraph_single_modules_coomatrix_rel_coords(community_vect,node_coords,Z_list,node_labels.tolist())
    #coclass_all_modules_files = plot_3D_igraph_all_modules_coomatrix_rel_coords(community_vect,node_coords,Z_list,node_labels.tolist())
    
    
    #return coclass_single_modules_files,coclass_all_modules_files
    
    
#def plot_igraph_modules_node_roles_rada(rada_lol_file,Pajek_net_file,coords_file,net_List_file,gm_mask_coords_file,node_roles_file):

    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes,read_List_net_file
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_modules_node_roles

    #print 'Loading node_corres'
    
    #node_corres = read_Pajek_corres_nodes(Pajek_net_file)
    
    #print node_corres
    #print node_corres.shape
    
    #print 'Loading coords'
    
    
    ##with open(coords_file, 'Ur') as f:
        ##coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    #print 'Loading gm mask coords'
    
    #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    #print gm_mask_coords.shape
    
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    #print community_vect.shape
    
    #print "loading net_List_net as list"
    
    #Z_list = read_List_net_file(net_List_file)
    
    #print Z_list
    
    #print 'extracting node coords'
    
    #node_coords = coords[node_corres,:]
    
    #print node_coords.shape
    
    #print "reading node roles"
    
    #node_roles = np.array(np.loadtxt(node_roles_file),dtype = 'int64')
    
    #print node_roles
    
    
    
    
    #print "plotting conf_cor_mat_modules_file with igraph"
    
    #plot_igraph_node_roles_file = plot_3D_igraph_modules_node_roles(community_vect,node_coords,Z_list,gm_mask_coords,node_roles)
    
    #return plot_igraph_node_roles_file
    
#def plot_igraph_modules_coclass_rada_forced_colors(rada_lol_file,Pajek_net_file,gm_mask_coords_file,labels_file,rois_orig_indexes_file):
    
    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix,read_List_net_file,force_order
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_single_modules_coomatrix_rel_coords,plot_3D_igraph_all_modules_coomatrix_rel_coords
##    from dmgraphanalysis.plot_igraph import plot_3D_igraph_modules_net_list

    #print 'loading labels'
    
    #labels = [line.strip() for line in open(labels_file)]
    
    #print 'loading rois_orig_indexes_file'
    
    #rois_orig_indexes = np.array(np.loadtxt(rois_orig_indexes_file),dtype = int)
    
    #print rois_orig_indexes.shape
    
    #print 'Loading node_corres'
    
    #node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    
    ##print np.min(node_corres),np.max(node_corres)
    ##print node_corres.shape
    
    ##print Z_list.shape
    #print 'Loading gm mask coords'
    
    #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    ##print gm_mask_coords.shape
    
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    #print community_vect.shape
    
    #print "loading net_list_net as list"
    
    ##net_list = read_List_net_file(net_list_file)
    
    ##print net_list
    
    #print 'extracting node coords'
    
    #node_coords = gm_mask_coords[node_corres,:]
    
    #print 'extracting node orig_indexes'
    
    #node_orig_indexes = rois_orig_indexes[node_corres,:]
    
    #print node_orig_indexes
    
    #reordered_community_vect = force_order(community_vect,node_orig_indexes)
    
    #print reordered_community_vect.shape
        
    #np_labels = np.array(labels,dtype = 'string')
    
    #node_labels = np_labels[node_corres,:]
    
    ##print node_labels.shape
    
    ##print len(labels),node_labels.shape
    
    
    #print "plotting Z_cor_mat_modules_file with igraph"
    
    #coclass_single_modules_files = plot_3D_igraph_single_modules_coomatrix_rel_coords(reordered_community_vect,node_coords,Z_list,node_labels.tolist())
    #coclass_all_modules_files = plot_3D_igraph_all_modules_coomatrix_rel_coords(reordered_community_vect,node_coords,Z_list,node_labels.tolist())
    
    #return coclass_single_modules_files,coclass_all_modules_files
    
    
    ##### special RS-monkey, where coords are forced in Pajek file, and used to place nodes on plot
#def plot_igraph_modules_read_pajek_rel_coords_rada(rada_lol_file,Pajek_net_file,coords_net_file):

    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix,read_Pajek_rel_coords
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_single_modules_coomatrix_rel_coords,plot_3D_igraph_all_modules_coomatrix_rel_coords

    #print 'Loading node_corres and Z list'
    
    #node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    #print np.min(node_corres),np.max(node_corres)
    
    #print node_corres.shape
    
    ##print Z_list 
    ##print Z_list.shape 
    
    #print 'Loading coords'
    
    #node_rel_coords = read_Pajek_rel_coords(coords_net_file)
    
    #print node_rel_coords
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    #print community_vect.shape
    #print np.min(community_vect),np.max(community_vect)
    
    #print "plotting conf_cor_mat_modules_file with igraph"
    
    #Z_list_single_modules_files = plot_3D_igraph_single_modules_coomatrix_rel_coords(community_vect,node_rel_coords,Z_list)
    #Z_list_all_modules_files = plot_3D_igraph_all_modules_coomatrix_rel_coords(community_vect,node_rel_coords,Z_list)
    
    #return Z_list_single_modules_files,Z_list_all_modules_files
    
    
    
    
#def plot_igraph_modules_read_pajek_rel_coords_node_roles_rada(rada_lol_file,Pajek_net_file,coords_net_file,node_roles_file):

    #import numpy as np
    #import nibabel as nib
    #import os
    #import csv
        
    #from dmgraphanalysis.utils_cor import return_mod_mask_corres,read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix,read_Pajek_rel_coords
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_all_modules_coomatrix_rel_coords_node_roles,plot_3D_igraph_single_modules_coomatrix_rel_coords_node_roles

    #print 'Loading node_corres and Z list'
    
    #node_corres,Z_list = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    #print np.min(node_corres),np.max(node_corres)
    
    #print node_corres.shape
    
    ##print Z_list 
    ##print Z_list.shape 
    
    #print 'Loading coords'
    
    #node_rel_coords = read_Pajek_rel_coords(coords_net_file)
    
    #print node_rel_coords
    
    #print "Loading community belonging file" + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    ##print community_vect
    #print community_vect.shape
    #print np.min(community_vect),np.max(community_vect)
    
    
    #print "Loading node roles"
    
    #node_roles = np.array(np.loadtxt(node_roles_file),dtype = 'int64')
    
    #print node_roles
    
    #print "plotting conf_cor_mat_modules_file with igraph"
    
    #node_roles_all_modules_files = plot_3D_igraph_all_modules_coomatrix_rel_coords_node_roles(community_vect,node_rel_coords,Z_list,node_roles)
    
    #node_roles_single_modules_files = plot_3D_igraph_single_modules_coomatrix_rel_coords_node_roles(community_vect,node_rel_coords,Z_list,node_roles,nb_min_nodes_by_module = 5)
    
    #return node_roles_single_modules_files,node_roles_all_modules_files
    
    
    
#def plot_igraph_matrix(mod_cor_mat_file,mod_average_coords_file):

    #import os
    ##import igraph as ig
    #import numpy as np
    
    #from dmgraphanalysis.plot_igraph import plot_3D_igraph_weighted_signed_matrix
    
    #print 'loading module (node) coordinates'
    
    ##mod_average_coords = np.loadtxt(mod_average_coords_file)
    

    #print 'load coords'
    
    #mod_average_coords = np.loadtxt(mod_average_coords_file)
    
    
    ##with open(mod_average_coords_file, 'Ur') as f:
        ##mod_average_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    ##print mod_average_coords
    
    #print "loading mod cor mat"
    
    #mod_cor_mat = np.loadtxt(mod_cor_mat_file)
    
    #i_graph_file = plot_3D_igraph_weighted_signed_matrix(mod_cor_mat,mod_average_coords)
    
    #return i_graph_file
    
################################################################# Node roles 

#def compute_node_role_rada(rada_lol_file,Pajek_net_file):

    #import numpy as np
    
    #import nibabel as nib
    #import os
    #from dmgraphanalysis.utils_cor import read_lol_file,read_Pajek_corres_nodes_and_sparse_matrix,compute_roles

    #print 'Loading Pajek_net_file for reading node_corres'
    
    #node_corres,sparse_mat = read_Pajek_corres_nodes_and_sparse_matrix(Pajek_net_file)
    
    #print sparse_mat.todense()
    
    #print node_corres.shape,sparse_mat.todense().shape
    
    #print "Loading community belonging file " + rada_lol_file

    #community_vect = read_lol_file(rada_lol_file)
    
    #print community_vect
    
    #print "Computing node roles"
    #node_roles,all_Z_com_degree,all_participation_coeff = compute_roles(community_vect,sparse_mat)
    
    #print node_roles
    
    
    #node_roles_file = os.path.abspath('node_roles.txt')
    
    #np.savetxt(node_roles_file,node_roles,fmt = '%d')
    
    
    #all_Z_com_degree_file = os.path.abspath('all_Z_com_degree.txt')
    
    #np.savetxt(all_Z_com_degree_file,all_Z_com_degree,fmt = '%f')
    
    
    #all_participation_coeff_file = os.path.abspath('all_participation_coeff.txt')
    
    #np.savetxt(all_participation_coeff_file,all_participation_coeff,fmt = '%f')
    
    #return node_roles_file,all_Z_com_degree_file,all_participation_coeff_file
    