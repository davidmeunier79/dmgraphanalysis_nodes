# -*- coding: utf-8 -*-

"""
Definition of nodes for computing reordering and plotting coclass_matrices
"""
import numpy as np
import os

from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined
    
############################################################################################### PrepareCoclass #####################################################################################################

from dmgraphanalysis_nodes.utils_cor import return_coclass_mat
#,return_hierachical_order
from dmgraphanalysis_nodes.utils_net import read_Pajek_corres_nodes,read_lol_file

class PrepareCoclassInputSpec(BaseInterfaceInputSpec):
    
    mod_files = traits.List(File(exists=True), desc='list of all files representing modularity assignement (in rada, lol files) for each subject', mandatory=True)
    
    coords_files = traits.List(File(exists=True), desc='list of all coordinates in numpy space files (in txt format) for each subject (after removal of non void data)', mandatory=True)
    
    node_corres_files = traits.List(File(exists=True), desc='list of all Pajek files (in txt format) to extract correspondance between nodes in rada analysis and original subject coordinates for each subject (as obtained from PrepRada)', mandatory=True)
    
    gm_mask_coords_file = File(exists=True, desc='Coordinates in numpy space, corresponding to all possible nodes in the original space', mandatory=True)
    
    
class PrepareCoclassOutputSpec(TraitedSpec):
    
    group_coclass_matrix_file = File(exists=True, desc="all coclass matrices of the group in .npy (pickle format)")
    
    sum_coclass_matrix_file = File(exists=True, desc="sum of coclass matrix of the group in .npy (pickle format)")
    
    sum_possible_edge_matrix_file = File(exists=True, desc="sum of possible edges matrices of the group in .npy (pickle format)")
    
    norm_coclass_matrix_file = File(exists=True, desc="sum of coclass matrix normalized by possible edges matrix of the group in .npy (pickle format)")
    
    
    
class PrepareCoclass(BaseInterface):
    
    """
    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1
    """
    input_spec = PrepareCoclassInputSpec
    output_spec = PrepareCoclassOutputSpec

    def _run_interface(self, runtime):
                
        print 'in prepare_coclass'
        mod_files = self.inputs.mod_files
        coords_files = self.inputs.coords_files
        node_corres_files = self.inputs.node_corres_files
        gm_mask_coords_file = self.inputs.gm_mask_coords_file
        
        print 'loading gm mask corres'
        
        gm_mask_coords = np.loadtxt(gm_mask_coords_file)
        
        print gm_mask_coords.shape
            
        #### read matrix from the first group
        #print Z_cor_mat_files
        
        sum_coclass_matrix = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        sum_possible_edge_matrix = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        
        #print sum_coclass_matrix.shape
        
                
        group_coclass_matrix = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0],len(mod_files)),dtype = float)
        
        print group_coclass_matrix.shape
        
        if len(mod_files) != len(coords_files) or len(mod_files) != len(node_corres_files):
            print "warning, length of mod_files, coords_files and node_corres_files are imcompatible {} {} {}".format(len(mod_files),len(coords_files),len(node_corres_files))
        
        for index_file in range(len(mod_files)):
        #for index_file in range(1):
                
            print mod_files[index_file]
            
            if os.path.exists(mod_files[index_file]) and os.path.exists(node_corres_files[index_file]) and os.path.exists(coords_files[index_file]):
            
                community_vect = read_lol_file(mod_files[index_file])
                print "community_vect:"
                print community_vect.shape
                
                node_corres_vect = read_Pajek_corres_nodes(node_corres_files[index_file])
                print "node_corres_vect:"
                print node_corres_vect.shape
                
                
                coords = np.loadtxt(coords_files[index_file])
                print "coords_subj:"
                print coords.shape
                
                
                corres_coords = coords[node_corres_vect,:]
                print "corres_coords:"
                print corres_coords.shape
                
                
                coclass_mat,possible_edge_mat = return_coclass_mat(community_vect,corres_coords,gm_mask_coords)
                
                np.fill_diagonal(coclass_mat,0)
                
                np.fill_diagonal(possible_edge_mat,1)
                
                sum_coclass_matrix += coclass_mat
                
                sum_possible_edge_matrix += possible_edge_mat
                
                group_coclass_matrix[:,:,index_file] = coclass_mat
                
                
            else:
                print "Warning, one or more files between " + mod_files[index_file] + ',' + node_corres_files[index_file] + ', ' + coords_files[index_file] + " do not exists"
            
            
        group_coclass_matrix_file= os.path.abspath('group_coclass_matrix.npy')
        
        np.save(group_coclass_matrix_file,group_coclass_matrix)
        
            
        print 'saving coclass matrix'
        
        sum_coclass_matrix_file = os.path.abspath('sum_coclass_matrix.npy')
        
        np.save(sum_coclass_matrix_file,sum_coclass_matrix)
        
        print 'saving possible_edge matrix'
        
        sum_possible_edge_matrix_file = os.path.abspath('sum_possible_edge_matrix.npy')
        
        np.save(sum_possible_edge_matrix_file,sum_possible_edge_matrix)
        
        
        #### save norm_coclass_matrix
        print 
        
        norm_coclass_matrix = np.zeros((gm_mask_coords.shape[0],gm_mask_coords.shape[0]),dtype = int)
        
        print np.where(np.array(sum_possible_edge_matrix == 0))
            
        norm_coclass_matrix = np.divide(np.array(sum_coclass_matrix,dtype = float),np.array(sum_possible_edge_matrix,dtype = float)) * 100
        
        
        #0/0
        
        print 'saving norm coclass matrix'
        
        norm_coclass_matrix_file =  os.path.abspath('norm_coclass_matrix.npy')
        
        np.save(norm_coclass_matrix_file,norm_coclass_matrix)
        
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["group_coclass_matrix_file"] = os.path.abspath('group_coclass_matrix.npy')
        
        outputs["sum_coclass_matrix_file"] = os.path.abspath('sum_coclass_matrix.npy')
        
        outputs["sum_possible_edge_matrix_file"] = os.path.abspath('sum_possible_edge_matrix.npy')
        
        outputs["norm_coclass_matrix_file"] =  os.path.abspath('norm_coclass_matrix.npy')
        
        return outputs
        
        ############################################################################################### DiffMatrices #####################################################################################################

from dmgraphanalysis_nodes.utils import check_np_shapes
        
class DiffMatricesInputSpec(BaseInterfaceInputSpec):
    
    mat_file1 = File(exists=True, desc='Matrix in npy format', mandatory=True)
    
    mat_file2 = File(exists=True, desc='Matrix in npy format', mandatory=True)
    
    
class DiffMatricesOutputSpec(TraitedSpec):
    
    diff_mat_file = File(exists=True, desc='Difference of Matrices (mat1 - mat2) in npy format', mandatory=True)
    
class DiffMatrices(BaseInterface):
    
    """
    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1
    """
    input_spec = DiffMatricesInputSpec
    output_spec = DiffMatricesOutputSpec

    def _run_interface(self, runtime):
                
        print 'in prepare_coclass'
        mat_file1 = self.inputs.mat_file1
        mat_file2 = self.inputs.mat_file2
            
        mat1 = np.load(mat_file1)
        print mat1.shape
        
        mat2 = np.load(mat_file2)
        print mat2.shape
        
        if check_np_shapes(mat1.shape,mat2.shape):
            
            diff_mat = mat1 - mat2
            print diff_mat
            
            diff_mat_file = os.path.abspath("diff_matrix.npy")
            
            np.save(diff_mat_file,diff_mat)
            
        else:
        
            print "Warning, shapes are different, cannot substrat matrices"
            sys.exit()
            
        
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["diff_mat_file"] = os.path.abspath("diff_matrix.npy")
        
        return outputs
        
#def diff_matrix(mat_file1,mat_file2):

        
############################################################################################### PlotCoclass #####################################################################################################

from nipype.utils.filemanip import split_filename as split_f

from dmgraphanalysis_nodes.utils_plot import plot_ranged_cormat

class PlotCoclassInputSpec(BaseInterfaceInputSpec):
    
    coclass_matrix_file = File(exists=True,  desc='coclass matrix in npy format', mandatory=True)
    
    labels_file = File(exists=True,  desc='labels of nodes', mandatory=False)
    
    list_value_range = traits.ListInt(desc='force the range of the plot', mandatory=False)
    
class PlotCoclassOutputSpec(TraitedSpec):
    
    plot_coclass_matrix_file = File(exists=True, desc="eps file with graphical representation")
    
class PlotCoclass(BaseInterface):
    
    """
    Plot coclass matrix 
    - labels are optional
    - range values are optional (default is min and max values of the matrix)
    """
    input_spec = PlotCoclassInputSpec
    output_spec = PlotCoclassOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_coclass'
        
        coclass_matrix_file = self.inputs.coclass_matrix_file
        labels_file = self.inputs.labels_file
        list_value_range = self.inputs.list_value_range
            
        
        print 'loading coclass'
        coclass_mat = np.load(coclass_matrix_file)
        
        
        if isdefined(labels_file):
            
            print 'loading labels'
            labels = [line.strip() for line in open(labels_file)]
            
        else :
            labels = []
            
        if not isdefined(list_value_range):
        
            list_value_range = [np.amin(coclass_mat),np.amax(coclass_mat)]
        
        print 'plotting heatmap'
        
        path,fname,ext = split_f(coclass_matrix_file)
        
        plot_coclass_matrix_file =  os.path.abspath('heatmap_' + fname + '.eps')
        
        plot_ranged_cormat(plot_coclass_matrix_file,coclass_mat,labels,fix_full_range = list_value_range)
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        path,fname,ext = split_f(self.inputs.coclass_matrix_file)
        
        outputs["plot_coclass_matrix_file"] = os.path.abspath('heatmap_' + fname + '.eps')
        
        return outputs
        
############################################################################################### PlotIGraphCoclass #####################################################################################################

from nipype.utils.filemanip import split_filename as split_f

from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
    
class PlotIGraphCoclassInputSpec(BaseInterfaceInputSpec):
    
    coclass_matrix_file = File(exists=True,  desc='coclass matrix in npy format', mandatory=True)
    
    labels_file = File(exists=True,  desc='labels of nodes', mandatory=False)
    
    threshold = traits.Int(50, usedefault = True, desc='What min coclass value is reresented by an edge on the graph', mandatory=False)
    
    gm_mask_coords_file = File(exists=True,  desc='node coordiantes in MNI space (txt file)', mandatory=False)
    
class PlotIGraphCoclassOutputSpec(TraitedSpec):
    
    plot_igraph_3D_coclass_matrix_file = File(exists=True, desc="eps file with igraph graphical representation")
    
class PlotIGraphCoclass(BaseInterface):
    
    """
    Plot coclassification matrix with igraph
    - labels are optional, 
    - threshold is optional (default, 50 = half the group)
    - coordinates are optional, if no coordiantes are specified, representation in topological (Fruchterman-Reingold) space
    """
    input_spec = PlotIGraphCoclassInputSpec
    output_spec = PlotIGraphCoclassOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_coclass'
        
        coclass_matrix_file = self.inputs.coclass_matrix_file
        labels_file = self.inputs.labels_file
        
        threshold = self.inputs.threshold
        gm_mask_coords_file = self.inputs.gm_mask_coords_file
        
        
        print 'loading coclass'
        coclass_matrix = np.load(coclass_matrix_file)
        
        
        if isdefined(labels_file):
            
            print 'loading labels'
            labels = [line.strip() for line in open(labels_file)]
            
        else :
            labels = []
            
        if isdefined(gm_mask_coords_file):
            
            print 'loading coords'
            
            gm_mask_coords = np.loadtxt(gm_mask_coords_file)
            
        else :
            gm_mask_coords = np.array([])
            
            
        print 'thresholding coclass'
        
        coclass_matrix[coclass_matrix <= threshold] = 0
        
        coclass_matrix[coclass_matrix > threshold] = 1
        
        print coclass_matrix
        
        print 'plotting igraph'
        
        
        plot_igraph_3D_coclass_matrix_file = os.path.abspath("plot_3D_signif_coclass_mat.eps")
        
        plot_3D_igraph_int_mat(plot_igraph_3D_coclass_matrix_file,coclass_matrix,coords = gm_mask_coords,labels = labels)
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["plot_igraph_3D_coclass_matrix_file"] = os.path.abspath("plot_3D_signif_coclass_mat.eps")
        
        return outputs
        
        
############################################################################################### PlotIGraphConjCoclass #####################################################################################################

from dmgraphanalysis_nodes.plot_igraph import plot_3D_igraph_int_mat
from dmgraphanalysis_nodes.utils import check_np_shapes

class PlotIGraphConjCoclassInputSpec(BaseInterfaceInputSpec):
    
    coclass_matrix_file1 = File(exists=True,  desc='coclass matrix in npy format', mandatory=True)
    coclass_matrix_file2 = File(exists=True,  desc='coclass matrix in npy format', mandatory=True)
    
    labels_file = File(exists=True,  desc='labels of nodes', mandatory=False)
    threshold = traits.Int(50, usedefault = True, desc='What min coclass value is reresented by an edge on the graph', mandatory=False)
    gm_mask_coords_file = File(exists=True,  desc='node coordiantes in MNI space (txt file)', mandatory=False)
    
class PlotIGraphConjCoclassOutputSpec(TraitedSpec):
    
    plot_igraph_conj_signif_coclass_matrix_file = File(exists=True, desc="eps file with igraph spatial representation")
    plot_igraph_FR_conj_signif_coclass_matrix_file = File(exists=True, desc="eps file with igraph topological representation")
    
class PlotIGraphConjCoclass(BaseInterface):
    
    """
    Plot coclassification matrix with igraph
    - labels are optional, 
    - threshold is optional (default, 50 = half the group)
    - coordinates are optional, if no coordiantes are specified, representation in topological (Fruchterman-Reingold) space
    """
    input_spec = PlotIGraphConjCoclassInputSpec
    output_spec = PlotIGraphConjCoclassOutputSpec

    def _run_interface(self, runtime):
                
        print 'in plot_coclass'
        
        coclass_matrix_file1 = self.inputs.coclass_matrix_file1
        coclass_matrix_file2 = self.inputs.coclass_matrix_file2
        labels_file = self.inputs.labels_file
        
        threshold = self.inputs.threshold
        gm_mask_coords_file = self.inputs.gm_mask_coords_file
            
        #from dmgraphanalysis.utils_plot import plot_ranged_cormat
        
        
        #from nipype.utils.filemanip import split_filename as split_f
        
        print 'loading labels'
        
        labels = [line.strip() for line in open(labels_file)]
        
        
        print 'loading coclass_matrices'
        coclass_matrix1 = np.load(coclass_matrix_file1)
        coclass_matrix2 = np.load(coclass_matrix_file2)
        
        path,fname,ext = split_f(coclass_matrix_file1)
        
        
        print 'loading gm mask corres'
        
        gm_mask_coords = np.array(np.loadtxt(gm_mask_coords_file),dtype = 'float')
        
        print gm_mask_coords.shape
            
            
        print 'computing diff coclass'
        
        if not check_np_shapes(coclass_matrix1.shape,coclass_matrix2.shape):
            
            print "$$$$$$$$ exiting, unequal shapes for coclass matrices"
            
            sys.exit()
            
        diff_matrix = coclass_matrix1 - coclass_matrix2
        
        #### 
        print "plotting diff matrix"    
        
        plot_diff_coclass_matrix_file =  os.path.abspath('heatmap_diff_coclass_matrix.eps')
        
        plot_ranged_cormat(plot_diff_coclass_matrix_file,diff_matrix,labels,fix_full_range = [-50,50])
        
        
        
        print "separating the overlap and signif diff netwtorks"
        
        conj_labelled_matrix = np.zeros(shape = diff_matrix.shape, dtype = 'int')
        
        conj_labelled_matrix[np.logical_and(coclass_matrix1 > threshold,coclass_matrix2 > threshold)] = 1
        
        if  np.sum(conj_labelled_matrix != 0) != 0:
                
            plot_igraph_conj_coclass_matrix_file = os.path.abspath('plot_igraph_3D_conj_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_conj_coclass_matrix_file,conj_labelled_matrix,gm_mask_coords,labels = labels)
            
            plot_igraph_FR_conj_coclass_matrix_file = os.path.abspath('plot_igraph_FR_conj_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_FR_conj_coclass_matrix_file,conj_labelled_matrix,labels = labels)
            
        ## signif coclass1
        
        signif_coclass1_labelled_matrix = np.zeros(shape = diff_matrix.shape, dtype = 'int')
        
        signif_coclass1_labelled_matrix[np.logical_and(coclass_matrix1 > threshold,diff_matrix > 25)] = 1
        
        if np.sum(signif_coclass1_labelled_matrix != 0) != 0:
            
            plot_igraph_signif_coclass1_coclass_matrix_file = os.path.abspath('plot_igraph_3D_signif_coclass1_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_signif_coclass1_coclass_matrix_file,signif_coclass1_labelled_matrix,gm_mask_coords,labels = labels)
            
            plot_igraph_FR_signif_coclass1_coclass_matrix_file = os.path.abspath('plot_igraph_FR_signif_coclass1_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_FR_signif_coclass1_coclass_matrix_file,signif_coclass1_labelled_matrix,labels = labels)
            
        
        ## signif coclass2
        
        signif_coclass2_labelled_matrix = np.zeros(shape = diff_matrix.shape, dtype = 'int')
        
        signif_coclass2_labelled_matrix[np.logical_and(coclass_matrix2 > threshold,diff_matrix < -25)] = 1
        
        
        if np.sum(signif_coclass2_labelled_matrix != 0) != 0:
        
            plot_igraph_signif_coclass2_coclass_matrix_file = os.path.abspath('plot_igraph_3D_signif_coclass2_coclass_matrix.eps')
        
            plot_3D_igraph_int_mat(plot_igraph_signif_coclass2_coclass_matrix_file,signif_coclass2_labelled_matrix,gm_mask_coords,labels = labels)
        
            plot_igraph_FR_signif_coclass2_coclass_matrix_file = os.path.abspath('plot_igraph_FR_signif_coclass2_coclass_matrix.eps')
        
            plot_3D_igraph_int_mat(plot_igraph_FR_signif_coclass2_coclass_matrix_file,signif_coclass2_labelled_matrix,labels = labels)
        
        
        print "computing signif int_labelled_signif_matrix"
            
        int_labelled_signif_matrix = np.zeros(shape = diff_matrix.shape, dtype = 'int')
        
        #int_labelled_signif_matrix[np.logical_and(coclass_matrix1 > threshold,coclass_matrix2 > threshold)] = 1
        
        #int_labelled_signif_matrix[diff_matrix > 50] = 2
        #int_labelled_signif_matrix[-diff_matrix < -50] = 3
        
        int_labelled_signif_matrix[conj_labelled_matrix == 1] = 1
        
        
        int_labelled_signif_matrix[signif_coclass1_labelled_matrix == 1] = 2
        int_labelled_signif_matrix[signif_coclass2_labelled_matrix == 1] = 3
        
        
        print int_labelled_signif_matrix
        
        print 'plotting igraph'
        
        if np.sum(int_labelled_signif_matrix != 0) != 0:
                
            plot_igraph_conj_signif_coclass_matrix_file = os.path.abspath('plot_igraph_3D_conj_signif_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_conj_signif_coclass_matrix_file,int_labelled_signif_matrix,gm_mask_coords,labels = labels)
            
        
            plot_igraph_FR_conj_signif_coclass_matrix_file = os.path.abspath('plot_igraph_FR_conj_signif_coclass_matrix.eps')
            
            plot_3D_igraph_int_mat(plot_igraph_FR_conj_signif_coclass_matrix_file,int_labelled_signif_matrix,labels = labels)
            
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["plot_igraph_conj_signif_coclass_matrix_file"] = os.path.abspath('plot_igraph_3D_conj_signif_coclass_matrix.eps')
        outputs["plot_igraph_FR_conj_signif_coclass_matrix_file"] = os.path.abspath('plot_igraph_FR_conj_signif_coclass_matrix.eps')
        
        return outputs
        