# -*- coding: utf-8 -*-

import numpy as np
import os

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined
    

from nipype.utils.filemanip import split_filename as split_f
        
############################################################################################### StatsPairBinomial #####################################################################################################

import dmgraphanalysis_nodes.utils_stats as stats
        
class StatsPairBinomialInputSpec(BaseInterfaceInputSpec):
    
    group_coclass_matrix_file1 = File(exists=True,  desc='file of group 1 coclass matrices in npy format', mandatory=True)
    group_coclass_matrix_file2 = File(exists=True,  desc='file of group 2 coclass matrices in npy format', mandatory=True)
    
    conf_interval_binom_fdr = traits.Float(0.05, usedefault = True, desc='Alpha value used as FDR implementation', mandatory=False)
    
class StatsPairBinomialOutputSpec(TraitedSpec):
    
    signif_signed_adj_fdr_mat_file = File(exists=True, desc="int matrix with corresponding codes to significance")
    
class StatsPairBinomial(BaseInterface):
    
    """
    Plot coclassification matrix with igraph
    - labels are optional, 
    - threshold is optional (default, 50 = half the group)
    - coordinates are optional, if no coordiantes are specified, representation in topological (Fruchterman-Reingold) space
    """
    input_spec = StatsPairBinomialInputSpec
    output_spec = StatsPairBinomialOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_coclass'
        
        group_coclass_matrix_file1 = self.inputs.group_coclass_matrix_file1
        group_coclass_matrix_file2 = self.inputs.group_coclass_matrix_file2
        conf_interval_binom_fdr = self.inputs.conf_interval_binom_fdr
            

        print "loading group_coclass_matrix1"
        
        group_coclass_matrix1 = np.array(np.load(group_coclass_matrix_file1),dtype = float)
        print group_coclass_matrix1.shape
        
        
        print "loading group_coclass_matrix2"
        
        group_coclass_matrix2 = np.array(np.load(group_coclass_matrix_file2),dtype = float)
        print group_coclass_matrix2.shape
        
        
        print "compute NBS stats"
        
        
        # check input matrices
        Ix,Jx,nx = group_coclass_matrix1.shape
        Iy,Jy,ny = group_coclass_matrix2.shape
        
        assert Ix == Iy
        assert Jx == Jy
        assert Ix == Jx
        assert Iy == Jy
        
        signif_signed_adj_mat  = stats.compute_pairwise_binom_fdr(group_coclass_matrix1,group_coclass_matrix2,conf_interval_binom_fdr)
        
        print 'save pairwise signed stat file'
        
        signif_signed_adj_fdr_mat_file  = os.path.abspath('signif_signed_adj_fdr_'+ str(conf_interval_binom_fdr) +'.npy')
        np.save(signif_signed_adj_fdr_mat_file,signif_signed_adj_mat)
        
        #return signif_signed_adj_fdr_mat_file

            
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["signif_signed_adj_fdr_mat_file"] = os.path.abspath('signif_signed_adj_fdr_'+ str(self.inputs.conf_interval_binom_fdr) +'.npy')
        
        return outputs

############################################################################################### StatsPairTTest #####################################################################################################

import dmgraphanalysis_nodes.utils_stats as stats
        
class StatsPairTTestInputSpec(BaseInterfaceInputSpec):
    
    group_cormat_file1 = File(exists=True,  desc='file of group 1 cormat matrices in npy format', mandatory=True)
    group_cormat_file2 = File(exists=True,  desc='file of group 2 cormat matrices in npy format', mandatory=True)
    
    t_test_thresh_fdr = traits.Float(0.05, usedefault = True, desc='Alpha value used as FDR implementation', mandatory=False)
    
    paired = traits.Bool(True,usedefault = True, desc='Ttest is paired or not', mandatory=False)
    
class StatsPairTTestOutputSpec(TraitedSpec):
    
    signif_signed_adj_fdr_mat_file = File(exists=True, desc="int matrix with corresponding codes to significance")
    
class StatsPairTTest(BaseInterface):
    
    """
    Compute ttest stats between 2 group of matrix 
    - matrix are arranged in group_cormat, with order (Nx,Ny,Nsubj). Nx = Ny (each matricx is square)
    - t_test_thresh_fdr is optional (default, 0.05)
    - paired in indicate if ttest is pairde or not. If paired, both group have the same number of samples
    """
    input_spec = StatsPairTTestInputSpec
    output_spec = StatsPairTTestOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_cormat'
        
        group_cormat_file1 = self.inputs.group_cormat_file1
        group_cormat_file2 = self.inputs.group_cormat_file2
        t_test_thresh_fdr = self.inputs.t_test_thresh_fdr
            
        paired = self.inputs.paired
        print "loading group_cormat1"
        
        group_cormat1 = np.array(np.load(group_cormat_file1),dtype = float)
        print group_cormat1.shape
        
        
        print "loading group_cormat2"
        
        group_cormat2 = np.array(np.load(group_cormat_file2),dtype = float)
        print group_cormat2.shape
        
        
        print "compute NBS stats"
        
        
        # check input matrices
        Ix,Jx,nx = group_cormat1.shape
        Iy,Jy,ny = group_cormat2.shape
        
        assert Ix == Iy
        assert Jx == Jy
        assert Ix == Jx
        assert Iy == Jy
        
        signif_signed_adj_mat  = stats.compute_pairwise_ttest_fdr(group_cormat1,group_cormat2,t_test_thresh_fdr,paired)
        
        print 'save pairwise signed stat file'
        
        signif_signed_adj_fdr_mat_file  = os.path.abspath('signif_signed_adj_fdr_'+ str(t_test_thresh_fdr) +'.npy')
        np.save(signif_signed_adj_fdr_mat_file,signif_signed_adj_mat)
        
        #return signif_signed_adj_fdr_mat_file

            
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["signif_signed_adj_fdr_mat_file"] = os.path.abspath('signif_signed_adj_fdr_'+ str(self.inputs.t_test_thresh_fdr) +'.npy')
        
        return outputs

        
############################################################################################### PlotIGraphSignedIntMat #####################################################################################################

from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_signed_int_mat
from dmgraphanalysis_nodes.utils import check_np_shapes

#from dmgraphanalysis.utils_plot import plot_cormat
    
    
class PlotIGraphSignedIntMatInputSpec(BaseInterfaceInputSpec):
    
    signed_int_mat_file = File(exists=True,  desc='signed int matrix in npy format', mandatory=True)
    
    labels_file = File(exists=True,  desc='labels of nodes (txt file)', mandatory=False)
    coords_file = File(exists=True,  desc='node coordinates in MNI space (txt file)', mandatory=False)
    
class PlotIGraphSignedIntMatOutputSpec(TraitedSpec):
    
    plot_3D_signed_int_mat_file = File(exists=True, desc="eps file with igraph spatial representation")
    #plot_heatmap_signed_bin_mat_file = File(exists=True, desc="eps file heatmap representation")
    
class PlotIGraphSignedIntMat(BaseInterface):
    
    """
    Plot coclassification matrix with igraph
    - labels are optional, 
    - threshold is optional (default, 50 = half the group)
    - coordinates are optional, if no coordiantes are specified, representation in topological (Fruchterman-Reingold) space
    """
    input_spec = PlotIGraphSignedIntMatInputSpec
    output_spec = PlotIGraphSignedIntMatOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_coclass'
        
        signed_int_mat_file = self.inputs.signed_int_mat_file
        labels_file = self.inputs.labels_file
        coords_file = self.inputs.coords_file
            
            
        print 'load bin matrix'
        
        signed_int_mat = np.load(signed_int_mat_file)
        
        print signed_int_mat.shape
        
        if isdefined(labels_file):
            
            print 'loading labels'
            labels = [line.strip() for line in open(labels_file)]
            
        else :
            labels = []
            
        if isdefined(coords_file):
            
            print 'loading coords'
            coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
            
        else :
            coords = np.array([])
            
            
        print coords.shape
        
        
        print 'plotting igraph 3D'
        
        ######## igraph 3D
        plot_3D_signed_int_mat_file = os.path.abspath('plot_igraph_3D_signed_int_mat.eps')
            
        plot_3D_igraph_signed_int_mat(plot_3D_signed_int_mat_file,signed_int_mat,coords,labels)
        
                
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["plot_3D_signed_int_mat_file"] = os.path.abspath('plot_igraph_3D_signed_int_mat.eps')
        #outputs["plot_heatmap_signed_int_mat_file"] = os.path.abspath('heatmap_signed_bin_mat.eps')
        
        return outputs
        
        
        
        
############################################################################################### PrepareCormat #####################################################################################################

from dmgraphanalysis_nodes.utils_cor import return_corres_correl_mat
#,return_hierachical_order
        
class PrepareCormatInputSpec(BaseInterfaceInputSpec):
    
    cor_mat_files = traits.List(File(exists=True), desc='list of all correlation matrice files (in npy format) for each subject', mandatory=True)
    
    coords_files = traits.List(File(exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True)
    
    gm_mask_coords_file = File(exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=True)
    
    
class PrepareCormatOutputSpec(TraitedSpec):
    
    group_cormat_file = File(exists=True, desc="all cormat matrices of the group in .npy (pickle format)")
    
    avg_cormat_file = File(exists=True, desc="average of cormat matrix of the group in .npy (pickle format)")
    
    group_vect_file = File(exists=True, desc="degree (?) by nodes * indiv of the group in .npy (pickle format)")
    
    
class PrepareCormat(BaseInterface):
    
    """
    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1
    """
    input_spec = PrepareCormatInputSpec
    output_spec = PrepareCormatOutputSpec

    def _run_interface(self, runtime):
                
        print 'in prepare_coclass'
        cor_mat_files = self.inputs.cor_mat_files
        coords_files = self.inputs.coords_files
        gm_mask_coords_file = self.inputs.gm_mask_coords_file
        
        
    #import numpy as np
    #import os

    ##import nibabel as nib
        print 'loading gm mask corres'
        
        gm_mask_coords = np.loadtxt(gm_mask_coords_file)
        
        print gm_mask_coords.shape
            
        #### read matrix from the first group
        #print Z_cor_mat_files
        
        sum_cormat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = float)
        print sum_cormat.shape
        
                
        group_cormat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0],len(cor_mat_files)),dtype = float)
        print group_cormat.shape
        
        
        group_vect = np.zeros((gm_mask_coords.shape[0],len(cor_mat_files)),dtype = float)
        print group_vect.shape
        
        if len(cor_mat_files) != len(coords_files):
            print "warning, length of cor_mat_files, coords_files are imcompatible {} {} {}".format(len(cor_mat_files),len(coords_files))
        
        for index_file in range(len(cor_mat_files)):
            
            print cor_mat_files[index_file]
            
            if os.path.exists(cor_mat_files[index_file]) and os.path.exists(coords_files[index_file]):
            
                Z_cor_mat = np.load(cor_mat_files[index_file])
                print Z_cor_mat.shape
                
                
                coords = np.loadtxt(coords_files[index_file])
                print coords.shape
                
                
                
                corres_cor_mat,possible_edge_mat = return_corres_correl_mat(Z_cor_mat,coords,gm_mask_coords)
                
                print corres_cor_mat.shape
                print group_cormat.shape
                
                sum_cormat += corres_cor_mat
                
                group_cormat[:,:,index_file] = corres_cor_mat
                
                group_vect[:,index_file] = np.sum(corres_cor_mat,axis = 0)
                
                
            else:
                print "Warning, one or more files between " + cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists"
            
            
        group_cormat_file= os.path.abspath('group_cormat.npy')
        
        np.save(group_cormat_file,group_cormat)
        
            
        group_vect_file= os.path.abspath('group_vect.npy')
        
        np.save(group_vect_file,group_vect)
        
            
        print 'saving cor_mat matrix'
        
        avg_cormat_file = os.path.abspath('avg_cormat.npy')
        
        if (len(cor_mat_files) != 0):
        
                avg_cormat = sum_cormat /len(cor_mat_files)
                
                np.save(avg_cormat_file,avg_cormat)
        
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["group_cormat_file"] =os.path.abspath('group_cormat.npy')
        
        outputs["avg_cormat_file"] = os.path.abspath('avg_cormat.npy')
        
        outputs["group_vect_file"] = os.path.abspath('group_vect.npy')
        
        return outputs
        
        
        
        
        
#def prepare_nbs_stats_cor_mat_filter(cor_mat_files,coords_files,gm_mask_coords_file,filtered_nodes_file):
    
    #import numpy as np
    #import os

    ##import nibabel as nib
    
    #from dmgraphanalysis.utils_cor import return_corres_correl_mat
    ##from dmgraphanalysis.utils_cor import read_Pajek_corres_nodes,read_lol_file
    
    
    
    #print 'loading filtered nodes'
    
    
    #filtered_nodes = np.array(np.load(filtered_nodes_file),dtype = 'int')
    
    #print filtered_nodes
    
    #index_filtered_nodes, = np.where(filtered_nodes == 1)
    
    #print index_filtered_nodes.shape
    
    #print 'loading gm mask corres'
    
    #gm_mask_coords = np.loadtxt(gm_mask_coords_file)
    
    #print gm_mask_coords.shape
        
    ##### read matrix from the first group
    ##print Z_cor_mat_files
    
    #sum_cormat = np.zeros((index_filtered_nodes.shape[0],index_filtered_nodes.shape[0]),dtype = float)
    #print sum_cormat.shape
    
            
    #group_cormat = np.zeros((index_filtered_nodes.shape[0],index_filtered_nodes.shape[0],len(cor_mat_files)),dtype = float)
    #print group_cormat.shape
    
    
    #group_vect = np.zeros((index_filtered_nodes.shape[0],len(cor_mat_files)),dtype = float)
    #print group_vect.shape
    
    #if len(cor_mat_files) != len(coords_files):
        #print "warning, length of cor_mat_files, coords_files are imcompatible {} {} {}".format(len(cor_mat_files),len(coords_files))
    
    #for index_file in range(len(cor_mat_files)):
        
        #print cor_mat_files[index_file]
        
        #if os.path.exists(cor_mat_files[index_file]) and os.path.exists(coords_files[index_file]):
        
            #Z_cor_mat = np.load(cor_mat_files[index_file])
            #print Z_cor_mat.shape
            
            
            #coords = np.loadtxt(coords_files[index_file])
            #print coords.shape
            
            
            
            #corres_cor_mat,possible_edge_mat = return_corres_correl_mat(Z_cor_mat,coords,gm_mask_coords)
            
            #print corres_cor_mat.shape
            
            #tmp_filtered_corres_cor_mat = corres_cor_mat[index_filtered_nodes,:]
            #filtered_corres_cor_mat = tmp_filtered_corres_cor_mat[:,index_filtered_nodes]
            
            #print filtered_corres_cor_mat
            ##print filtered_corres_cor_mat.shape
            
            #print group_cormat.shape
            
            #sum_cormat += filtered_corres_cor_mat
            
            #group_cormat[:,:,index_file] = filtered_corres_cor_mat
            
            #group_vect[:,index_file] = np.sum(filtered_corres_cor_mat,axis = 0)
            
            
        #else:
            #print "Warning, one or more files between " + cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists"
        
        
    #group_cormat_file= os.path.abspath('group_cormat.npy')
    
    #np.save(group_cormat_file,group_cormat)
    
        
    #group_vect_file= os.path.abspath('group_vect.npy')
    
    #np.save(group_vect_file,group_vect)
    
        
    #print 'saving cor_mat matrix'
    
    #avg_cormat_file = os.path.abspath('avg_cormat.npy')
    
    #if (len(cor_mat_files) != 0):
    
            #avg_cormat = sum_cormat /len(cor_mat_files)
            
            #np.save(avg_cormat_file,avg_cormat)
    
    #return group_cormat_file,avg_cormat_file,group_vect_file
        
        
        
        
#def prepare_nbs_stats_cor_mat(cor_mat_files,coords_files,gm_mask_coords_file):
    
    #import numpy as np
    #import os

    ##import nibabel as nib
    
    #from dmgraphanalysis.utils_cor import return_corres_correl_mat
    ##from dmgraphanalysis.utils_cor import read_Pajek_corres_nodes,read_lol_file
    
    #print 'loading gm mask corres'
    
    #gm_mask_coords = np.loadtxt(gm_mask_coords_file)
    
    #print gm_mask_coords.shape
        
    ##### read matrix from the first group
    ##print Z_cor_mat_files
    
    #sum_cormat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = float)
    #print sum_cormat.shape
    
            
    #group_cormat = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0],len(cor_mat_files)),dtype = float)
    #print group_cormat.shape
    
    
    #group_vect = np.zeros((gm_mask_coords.shape[0],len(cor_mat_files)),dtype = float)
    #print group_vect.shape
    
    #if len(cor_mat_files) != len(coords_files):
        #print "warning, length of cor_mat_files, coords_files are imcompatible {} {} {}".format(len(cor_mat_files),len(coords_files))
    
    #for index_file in range(len(cor_mat_files)):
        
        #print cor_mat_files[index_file]
        
        #if os.path.exists(cor_mat_files[index_file]) and os.path.exists(coords_files[index_file]):
        
            #Z_cor_mat = np.load(cor_mat_files[index_file])
            #print Z_cor_mat.shape
            
            
            #coords = np.loadtxt(coords_files[index_file])
            #print coords.shape
            
            
            
            #corres_cor_mat,possible_edge_mat = return_corres_correl_mat(Z_cor_mat,coords,gm_mask_coords)
            
            #print corres_cor_mat.shape
            #print group_cormat.shape
            
            #sum_cormat += corres_cor_mat
            
            #group_cormat[:,:,index_file] = corres_cor_mat
            
            #group_vect[:,index_file] = np.sum(corres_cor_mat,axis = 0)
            
            
        #else:
            #print "Warning, one or more files between " + cor_mat_files[index_file] + ', ' + coords_files[index_file] + " do not exists"
        
        
    #group_cormat_file= os.path.abspath('group_cormat.npy')
    
    #np.save(group_cormat_file,group_cormat)
    
        
    #group_vect_file= os.path.abspath('group_vect.npy')
    
    #np.save(group_vect_file,group_vect)
    
        
    #print 'saving cor_mat matrix'
    
    #avg_cormat_file = os.path.abspath('avg_cormat.npy')
    
    #if (len(cor_mat_files) != 0):
    
            #avg_cormat = sum_cormat /len(cor_mat_files)
            
            #np.save(avg_cormat_file,avg_cormat)
    
    #return group_cormat_file,avg_cormat_file,group_vect_file
        
        
#def return_diff_group_mat(group_cormat_file1,group_cormat_file2):


    #import numpy as np
    #import os

    ##import nibabel as nib
    
    #from dmgraphanalysis.utils_cor import return_corres_correl_mat
    ##from dmgraphanalysis.utils_cor import read_Pajek_corres_nodes,read_lol_file
    
    #print 'loading group_cormat_file1 corres'
    
    #group_cormat1 = np.load(group_cormat_file1)
    
    #print group_cormat1.shape
    
    #print 'loading group_cormat_file2 corres'
    
    #group_cormat2 = np.load(group_cormat_file2)
    
    #print group_cormat2.shape
    
    #assert group_cormat1.shape == group_cormat2.shape
    
    #print 'computing difference between matrices group'
    
    #diff_group_cormat = group_cormat1 - group_cormat2
    
    #print diff_group_cormat.shape    
        
    #diff_group_cormat_file = os.path.abspath('diff_group_cormat.npy')
    
    #np.save(diff_group_cormat_file,diff_group_cormat)
        
    #return diff_group_cormat_file
        
        
########################################################################################################################################################################################################
#################################################################################### NBS stats #########################################################################################################
########################################################################################################################################################################################################

##def compute_coclass_rada_nbs_stats(group_coclass_matrix_file1,group_coclass_matrix_file2):
    
    ##import numpy as np
    ##import os

    ##import utils_nbs as nbs
    
    ##from  define_variables import THRESH,K,TAIL
    
    
    ##print "loading group_coclass_matrix1"
    
    ##group_coclass_matrix1 = np.array(np.load(group_coclass_matrix_file1),dtype = float)
    ##print group_coclass_matrix1.shape
    
    
    ##print "loading group_coclass_matrix2"
    
    ##group_coclass_matrix2 = np.array(np.load(group_coclass_matrix_file2),dtype = float)
    ##print group_coclass_matrix2.shape
    
    
    ##print "compute NBS stats"
    
    ##PVAL, ADJ, NULL, sz_links = nbs.compute_nbs(group_coclass_matrix1,group_coclass_matrix2,THRESH,K,TAIL)
    
    ###print 'nbs_stats_mat_file'
    
    ##print 'save nbs adj mat file' 
    ##nbs_adj_mat_file= os.path.abspath('nbs_adj_matrix_'+ str(THRESH) +'.npy')
    ##np.save(nbs_adj_mat_file,ADJ)
    
    ##print 'save nbs stat file'
    ##nbs_stats_file  = os.path.abspath('nbs_stats_'+ str(THRESH) +'.txt')
    ##np.savetxt(nbs_stats_file,PVAL,fmt = '%f')
    
    
    ##print 'save nbs NULL file'
    ##nbs_stats_file  = os.path.abspath('nbs_NULL_'+ str(THRESH) +'.txt')
    ##np.savetxt(nbs_stats_file,NULL,fmt = '%f')
    
    
    ##print 'save nbs sz_links file'
    ##nbs_stats_file  = os.path.abspath('nbs_sz_links_'+ str(THRESH) +'.txt')
    ##np.savetxt(nbs_stats_file,sz_links,fmt = '%f')
    
    
    
    
    ##return nbs_stats_file,nbs_adj_mat_file
    
    
##def plot_signif_nbs_adj_mat(nbs_adj_mat_file,gm_mask_coords_file,gm_mask_file):

    ##import os
    ##import numpy as np
    ##import nibabel as nib
    ##import csv
    
    ##from dmgraphanalysis.plot_igraph import plot_igraph_3D_int_label_mat
    
    ##from nipype.utils.filemanip import split_filename as split_f
    
    ##from dmgraphanalysis.utils_plot import plot_cormat
    
    ##from dmgraphanalysis.utils_cor import return_data_img_from_roi_mask
    
    ##print 'load adj matrix'
    
    ##signif_adj_matrix = np.load(nbs_adj_mat_file)
    
    ##print signif_adj_matrix.shape
    
    ##print 'load gm mask'
    
    ###with open(gm_mask_coords_file, 'Ur') as f:
        ###gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    ##gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    ##print gm_mask_coords.shape
    
    ###print gm_mask_coords
    
    ##print 'plotting igraph 3D'
    
    ########## igraph 3D
    ##plot_3D_nbs_adj_mat_file = os.path.abspath('plot_igraph_3D_signif_adj_mat.eps')
        
    ##plot_igraph_3D_int_label_mat(signif_adj_matrix,gm_mask_coords,plot_3D_nbs_adj_mat_file)
    
    ######### plot heat map
       
    ###### heatmap
    ##print 'plotting signif_adj_matrix heatmap'
    
    ##plot_heatmap_nbs_adj_mat_file =  os.path.abspath('heatmap_signif_adj_mat.eps')
    
    ##plot_cormat(plot_heatmap_nbs_adj_mat_file,signif_adj_matrix,list_labels = [])
    
    ###### significant coclassification degree in MNI space
    
    
    ##print 'format degree mask'
    
    ##signif_degree_img = return_data_img_from_roi_mask(gm_mask_file,np.sum(signif_adj_matrix,axis = 0))
    
    ##signif_degree_img.update_header()
    
    ##print 'saving degree mask'
    
    ##signif_degree_img_file = os.path.abspath('signif_degree.nii')
    
    ##nib.save(signif_degree_img,signif_degree_img_file)
    
    ##return plot_3D_nbs_adj_mat_file,plot_heatmap_nbs_adj_mat_file,signif_degree_img_file

##def plot_reorder_nbs_adj_matrix(nbs_adj_mat_file,node_order_vect_file,gm_mask_coords_file,gm_mask_file):
    
    ##import os
    ##import igraph as ig
    ##import numpy as np
    
    ##import csv
    
    ##print 'load adj matrix'
    
    ##signif_adj_matrix = np.load(nbs_adj_mat_file)
    
    ##print signif_adj_matrix.shape
    
    ##print 'load node_order_vect'
    
    ##node_order_vect = np.load(node_order_vect_file)
    
    ##print node_order_vect
    
    ##print "reorder signif_adj_matrix"
    
    
    ##signif_adj_matrix= signif_adj_matrix[node_order_vect,: ]
    
    ##reorder_signif_adj_matrix  = signif_adj_matrix[:, node_order_vect]
    
    
    ##print 'load gm mask'
    
    ##with open(gm_mask_coords_file, 'Ur') as f:
        ##gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    ###print gm_mask_coords
    
    
    
    ######### plot heat map
    
    ##import matplotlib.pyplot as plt
    ##import pylab as pl
    ##import nibabel as nib
    
    ##from nipype.utils.filemanip import split_filename as split_f
    
    
    ##print 'plotting heatmap'
    
    ##heatmap_reorder_nbs_adj_mat_file =  os.path.abspath('heatmap_reorder_signif_adj_mat.eps')
    
    ###fig1 = figure.Figure()
    ##fig1 = plt.figure()
    ##ax = fig1.add_subplot(1,1,1)
    ##im = ax.matshow(signif_adj_matrix)
    ##im.set_cmap('spectral')
    ##fig1.colorbar(im)
    
    ##fig1.savefig(heatmap_reorder_nbs_adj_mat_file)
    
    ##plt.close(fig1)
    ###fig1.close()
    ##del fig1
    
    ##return heatmap_reorder_nbs_adj_mat_file
    











        
########################################################################################################################################################################################################
############################################################################# pairwise/nodewise stats ##################################################################################################
########################################################################################################################################################################################################

######## plotting

#def plot_signed_bin_mat_labels_only_fdr(signed_bin_mat_file,coords_file,labels_file):

    #import os
    #import numpy as np
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.plot_igraph import plot_igraph_3D_signed_bin_label_mat
    
    #from dmgraphanalysis.utils_plot import plot_cormat
    
    #print 'loading labels'
    
    #labels = [line.strip() for line in open(labels_file)]
    
    
    #print 'load gm mask'
    
    ##with open(gm_mask_coords_file, 'Ur') as f:
        ##gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    
    #print 'load bin matrix'
    
    #signed_bin_mat = np.load(signed_bin_mat_file)
    
    #signed_bin_mat[np.abs(signed_bin_mat) < 1.5] = 0
    
    #print signed_bin_mat.shape
    
    #if np.sum(signed_bin_mat != 0) != 0:
        
        
        ##print gm_mask_coords
        
        #print 'plotting igraph 3D'
        
        ######### igraph 3D
        #plot_3D_signed_bin_mat_file = os.path.abspath('plot_igraph_3D_signed_bin_mat_only_fdr.eps')
            
        #plot_igraph_3D_signed_bin_label_mat(signed_bin_mat,coords,plot_3D_signed_bin_mat_file, labels = labels)
        
        ######## plot heat map
        
        ##### heatmap
        #print 'plotting signed_bin_mat heatmap'
        
        #plot_heatmap_signed_bin_mat_file =  os.path.abspath('heatmap_signed_bin_mat_only_fdr.eps')
        
        #plot_cormat(plot_heatmap_signed_bin_mat_file,signed_bin_mat,list_labels = labels)
        
        #return plot_3D_signed_bin_mat_file ,plot_heatmap_signed_bin_mat_file
    #else:
        
        #print "$$$$$$$$$$$$$$ Warning, Matrix is empty, no plotting"
        
        #return '' ,''
        
    
#def plot_signed_bin_mat_labels(signed_bin_mat_file,coords_file,labels_file):

    #import os
    #import numpy as np
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.plot_igraph import plot_igraph_3D_signed_bin_label_mat
    
    #from dmgraphanalysis.utils_plot import plot_cormat
    
    #print 'loading labels'
    
    #labels = [line.strip() for line in open(labels_file)]
    
    
    #print 'load bin matrix'
    
    #signed_bin_mat = np.load(signed_bin_mat_file)
    
    #print signed_bin_mat.shape
    
    #print 'load gm mask'
    
    ##with open(gm_mask_coords_file, 'Ur') as f:
        ##gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    ##print gm_mask_coords
    
    #print 'plotting igraph 3D'
    
    ######### igraph 3D
    #plot_3D_signed_bin_mat_file = os.path.abspath('plot_igraph_3D_signed_bin_mat.eps')
        
    #plot_igraph_3D_signed_bin_label_mat(signed_bin_mat,coords,plot_3D_signed_bin_mat_file, labels = labels)
    
    ######## plot heat map
       
    ##### heatmap
    #print 'plotting signed_bin_mat heatmap'
    
    #plot_heatmap_signed_bin_mat_file =  os.path.abspath('heatmap_signed_bin_mat.eps')
    
    #plot_cormat(plot_heatmap_signed_bin_mat_file,signed_bin_mat,list_labels = labels)
    
    #return plot_3D_signed_bin_mat_file ,plot_heatmap_signed_bin_mat_file
    
    
#def plot_signed_bin_mat(signed_bin_mat_file,coords_file):

    #import os
    #import numpy as np
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.plot_igraph import plot_igraph_3D_signed_bin_label_mat
    
    #from dmgraphanalysis.utils_plot import plot_cormat
    
    #print 'load bin matrix'
    
    #signed_bin_mat = np.load(signed_bin_mat_file)
    
    #print signed_bin_mat.shape
    
    #print 'load gm mask'
    
    ##with open(gm_mask_coords_file, 'Ur') as f:
        ##gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    ##print gm_mask_coords
    
    #print 'plotting igraph 3D'
    
    ######### igraph 3D
    #plot_3D_signed_bin_mat_file = os.path.abspath('plot_igraph_3D_signed_bin_mat.eps')
        
    #plot_igraph_3D_signed_bin_label_mat(signed_bin_mat,coords,plot_3D_signed_bin_mat_file)
    
    ######## plot heat map
       
    ##### heatmap
    #print 'plotting signed_bin_mat heatmap'
    
    #plot_heatmap_signed_bin_mat_file =  os.path.abspath('heatmap_signed_bin_mat.eps')
    
    #plot_cormat(plot_heatmap_signed_bin_mat_file,signed_bin_mat,list_labels = [])
    
    #return plot_3D_signed_bin_mat_file ,plot_heatmap_signed_bin_mat_file
    
#def plot_bin_mat(bin_mat_file,coords_file):

    #import os
    #import numpy as np
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.plot_igraph import plot_igraph_3D_int_label_mat
    
    #from dmgraphanalysis.utils_plot import plot_cormat
    
    #print 'load bin matrix'
    
    #bin_mat = np.load(bin_mat_file)
    
    #print bin_mat.shape
    
    #print 'load gm mask'
    
    ##with open(gm_mask_coords_file, 'Ur') as f:
        ##gm_mask_coords_list = list(tuple(map(float,rec))[0:2] for rec in csv.reader(f, delimiter=' '))
    
    #coords = np.array(np.loadtxt(coords_file),dtype = 'int64')
    
    #print coords.shape
    
    ##print gm_mask_coords
    
    #print 'plotting igraph 3D'
    
    ######### igraph 3D
    #plot_3D_bin_mat_file = os.path.abspath('plot_igraph_3D_bin_mat.eps')
        
    #plot_igraph_3D_int_label_mat(bin_mat,coords,plot_3D_bin_mat_file)
    
    ######## plot heat map
       
    ##### heatmap
    #print 'plotting bin_mat heatmap'
    
    #plot_heatmap_bin_mat_file =  os.path.abspath('heatmap_bin_mat.eps')
    
    #plot_cormat(plot_heatmap_bin_mat_file,bin_mat,list_labels = [])
    
    #return plot_3D_bin_mat_file ,plot_heatmap_bin_mat_file
    
    
#def plot_img_val_vect(val_vect_file,indexed_mask_file):

    #import os
    #import numpy as np
    #import nibabel as nib
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.utils_img import return_data_img_from_roi_mask
    
    #val_vect = np.load(val_vect_file)
    
    #print 'format degree mask'
    
    #val_vect_img = return_data_img_from_roi_mask(indexed_mask_file,val_vect)
    
    ##val_vect_img.update_header()
    
    #print 'saving val vect mask'
    
    #val_vect_img_file = os.path.abspath('val_vect_img.nii')
    
    #nib.save(val_vect_img,val_vect_img_file)
    
    #return val_vect_img_file
    
##################################### stats (using utils_stats)

    
#def compute_nodewise_ttest_stats_fdr(group_vect_file1,group_vect_file2,t_test_thresh_fdr):

    #import numpy as np
    #import os

    #import dmgraphanalysis.utils_stats as stats
    
    #print "loading group_vect1"
    
    #group_vect1 = np.array(np.load(group_vect_file1),dtype = float)
    #print group_vect1.shape
    
    
    #print "loading group_vect2"
    
    #group_vect2 = np.array(np.load(group_vect_file2),dtype = float)
    #print group_vect2.shape
    
    
    #print "compute NBS stats"
    
    
    ## check input matrices
    #Ix,nx = group_vect1.shape
    #Iy,ny = group_vect2.shape
    
    #assert Ix == Iy
    
    #nodewise_t_val_vect = stats.compute_nodewise_t_values_fdr(group_vect1,group_vect2,t_test_thresh_fdr)
    
    
    ##d_stacked = np.concatenate( (group_vect1,group_vect2),axis = 1)

    ##del group_vect1
    ##del group_vect2
    
    ##nodewise_t_val_vect = stats.compute_nodewise_t_test_vect(d_stacked,nx,ny)
    
    #print 'save nodewise stat file'
    ##pairwise_binom_adj_mat_file  = os.path.abspath('pairwise_binom_adj_'+ str(conf_interval_binom) +'.txt')
    ##np.savetxt(pairwise_binom_adj_mat_file,pairwise_binom_adj_mat,fmt = "%d")
    
    #nodewise_t_val_vect_file  = os.path.abspath('nodewise_t_val_vect_fdr_'+ str(t_test_thresh_fdr) +'.npy')
    #np.save(nodewise_t_val_vect_file,nodewise_t_val_vect)
    
    #return nodewise_t_val_vect_file

    
    ############################################################## correl with behav_score score ###############################################
    
#def compute_pairwise_correl_stats_fdr(group_cormat_file,behav_score,correl_thresh_fdr):

    #import numpy as np
    #import os

    #import dmgraphanalysis.utils_stats as stats
    
    #print "loading group_cormat1"
    
    #group_cormat = np.array(np.load(group_cormat_file),dtype = float)
    #print group_cormat.shape
    
    #assert group_cormat.shape[2] == len(behav_score)
    
    
    
    #signif_signed_adj_mat  = stats.compute_pairwise_correl_fdr(group_cormat,behav_score,correl_thresh_fdr)
    
    #print 'save pairwise signed stat file'
    
    #signif_signed_adj_fdr_mat_file  = os.path.abspath('signif_signed_adj_fdr_'+ str(correl_thresh_fdr) +'.npy')
    #np.save(signif_signed_adj_fdr_mat_file,signif_signed_adj_mat)
    
    #return signif_signed_adj_fdr_mat_file
    
#def remove_nodes_by_labels(remove_labelled_nodes,labels_file,gm_mask_coords_file):

    #import numpy as np
    #import os
    
    #from dmgraphanalysis.utils import get_multiple_indexes
    
    #from nipype.utils.filemanip import split_filename as split_f
    

    #print 'loading gm mask corres'
    
    #gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'int64')
    
    #print gm_mask_coords
    #print gm_mask_coords.shape
    
    #print 'loading labels'
    
    #labels = [line.strip() for line in open(labels_file)]
    
    #print labels
    
    ##### filtering nodes by labels
    #filtered_nodes = np.ones(shape = (len(labels)),dtype = int)
    
    #for remove_labelled_node in remove_labelled_nodes:
    
        #print remove_labelled_node
        
        #if remove_labelled_node in labels:
        
            #indexes = get_multiple_indexes(labels,remove_labelled_node)
            
            #print indexes
            
            #filtered_nodes[indexes] = 0
            
        ##print labels == remove_labelled_node 
    
    ##print [int(label != 
    
    #print filtered_nodes
    
    ##### filtered nodes
    #filtered_nodes_file = os.path.abspath("filtered_nodes.npy")
    
    #np.save(filtered_nodes_file,filtered_nodes)
    
    
    ##### read matrix from the first group
    ##print Z_cor_mat_files
    
    #index_filtered_nodes, = np.where(filtered_nodes == 1)
    
    #print index_filtered_nodes
    
    
        
    ##### filtered labels
    #filtered_labels = np.array(labels,dtype = "string")[index_filtered_nodes]
    
    #print filtered_labels.shape
    
    ##### saving filtered labels
    #path, fname, ext = split_f(labels_file)
    
    #filtered_labels_file = os.path.abspath(fname + '_filtered' + ext)
    
    #np.savetxt(filtered_labels_file,filtered_labels, fmt = '%s')
    
    
    
    ##### filtered coords
    #filtered_gm_mask_coords = gm_mask_coords[index_filtered_nodes,:]
    
    #print filtered_gm_mask_coords.shape
    
    ##### saving filtered coords 
    #path, fname, ext = split_f(gm_mask_coords_file)
    
    #filtered_gm_mask_coords_file = os.path.abspath(fname + '_filtered' + ext)
    
    #np.savetxt(filtered_gm_mask_coords_file,filtered_gm_mask_coords, fmt = '%d')
    
        
        
    #return filtered_nodes_file,filtered_labels_file,filtered_gm_mask_coords_file
    
    
    
    

    
