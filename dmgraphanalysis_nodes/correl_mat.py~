# -*- coding: utf-8 -*-

"""
Definition of Nodes for computing correlation matrices 
"""

#import nipy.labs.statistical_mapping as stat_map

#import itertools as iter
    
#import scipy.spatial.distance as dist
    
    
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined
    
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os

import nibabel as nib

from dmgraphanalysis_nodes.utils_plot import plot_signals,plot_sep_signals
from dmgraphanalysis_nodes.utils_cor import mean_select_mask_data
        
######################################################################################## ExtractTS ##################################################################################################################

class ExtractTSInputSpec(BaseInterfaceInputSpec):
    indexed_rois_file = File(exists=True, desc='indexed mask where all voxels belonging to the same ROI have the same value (! starting from 1)', mandatory=True)
    
    file_4D = File(exists=True, desc='4D volume to be extracted', mandatory=True)
    
    coord_rois_file = File(exists=True, desc='ROI coordinates', mandatory= False )
    
    min_BOLD_intensity = traits.Float(desc='BOLD signal below this value will be set to zero',mandatory=True)

class ExtractTSOutputSpec(TraitedSpec):
    
    mean_masked_ts_file = File(exists=True, desc="mean ts in .npy (pickle format)")
    
    subj_coord_rois_file = File(exists=True, desc="ROI coordinates retained for the subject")
    

class ExtractTS(BaseInterface):
    
    """Extract time series from a labelled mask in Nifti Format where all ROIs have the same index"""

    input_spec = ExtractTSInputSpec
    output_spec = ExtractTSOutputSpec

    def _run_interface(self, runtime):
            
        #import os
        #import numpy as np
        #import nibabel as nib
        
        #from dmgraphanalysis.utils_plot import plot_signals
        
        coord_rois_file = self.inputs.coord_rois_file
        indexed_rois_file = self.inputs.indexed_rois_file
        file_4D = self.inputs.file_4D
        min_BOLD_intensity = self.inputs.min_BOLD_intensity
        
        ## loading ROI coordinates
        coord_rois = np.loadtxt(coord_rois_file)
        
        print "coord_rois: " 
        print coord_rois.shape
        
        ## loading ROI indexed mask
        indexed_rois_img = nib.load(indexed_rois_file)
        
        indexed_mask_rois_data = indexed_rois_img.get_data()
        
        #print "indexed_mask_rois_data: "
        #print indexed_mask_rois_data.shape
        
        ### loading time series
        orig_ts = nib.load(file_4D).get_data()
        
        #print "orig_ts shape:"
        #print orig_ts.shape
            
        ### extrating ts by averaging the time series of all voxel with the same index
        sequence_roi_index = np.array(np.unique(indexed_mask_rois_data),dtype = int)
        
        if sequence_roi_index[0] == -1.0:
            sequence_roi_index = sequence_roi_index[1:]
        
        #print "sequence_roi_index:"
        #print sequence_roi_index
        
        if sequence_roi_index.shape[0] != coord_rois.shape[0]:
            print "Warning, indexes in template_ROI are incompatible with ROI coords"
            return
        
        mean_masked_ts = []
        subj_coord_rois = []
        
        for roi_index in sequence_roi_index:
        #for roi_index in sequence_roi_index[0:1]:
            
            #print "Roi index " + str(roi_index)
            
            index_roi_x,index_roi_y,index_roi_z = np.where(indexed_mask_rois_data == roi_index)
            
            #print index_roi_x,index_roi_y,index_roi_z
            
            all_voxel_roi_ts = orig_ts[index_roi_x,index_roi_y,index_roi_z,:]
            
            #print "all_voxel_roi_ts shape: "
            #print all_voxel_roi_ts.shape
            
            mean_roi_ts = np.mean(all_voxel_roi_ts,axis = 0)
            
            ### testing if mean_roi_ts if higher than minimal BOLD intensity
            
            if sum(mean_roi_ts > min_BOLD_intensity) == mean_roi_ts.shape[0]:
                
                #print "Roi selected: " + str(roi_index)
                #print "coord_rois shape: "
                #print coord_rois.shape
                
                subj_coord_rois.append(coord_rois[roi_index,])
                mean_masked_ts.append(mean_roi_ts)
            else:
                print "ROI " + str(roi_index) + " was not selected"
                
            
        mean_masked_ts = np.array(mean_masked_ts,dtype = 'f')
        subj_coord_rois = np.array(subj_coord_rois,dtype = 'float')
        
        print mean_masked_ts.shape
            
        ### saving time series
        mean_masked_ts_file = os.path.abspath("mean_masked_ts.txt")
        np.savetxt(mean_masked_ts_file,mean_masked_ts,fmt = '%.3f')
        
        ### saving subject ROIs
        subj_coord_rois_file = os.path.abspath("subj_coord_rois.txt")
        np.savetxt(subj_coord_rois_file,subj_coord_rois,fmt = '%.3f')
        
        
        print "plotting mean_masked_ts"
        
        plot_mean_masked_ts_file = os.path.abspath('mean_masked_ts.eps')    
        
        plot_signals(plot_mean_masked_ts_file,mean_masked_ts)
        
        return runtime
        
        #return mean_masked_ts_file,subj_coord_rois_file
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["mean_masked_ts_file"] = os.path.abspath("mean_masked_ts.txt")
        outputs["subj_coord_rois_file"] = os.path.abspath("subj_coord_rois.txt")
    
        return outputs

############################################################################################### ExtractMeanTS #####################################################################################################

class ExtractMeanTSInputSpec(BaseInterfaceInputSpec):
    mask_file = File(exists=True, desc='mask file where all voxels belonging to the selected region have index 1', mandatory=True)
    
    file_4D = File(exists=True, desc='4D volume to be extracted', mandatory=True)
    
    suffix = traits.String(desc='Suffix added to describe the extracted time series',mandatory=False)

    
class ExtractMeanTSOutputSpec(TraitedSpec):
    
    mean_masked_ts_file = File(exists=True, desc="mean ts in .npy (pickle format)")
    

class ExtractMeanTS(BaseInterface):
    
    """
    Extract mean time series from a labelled mask in Nifti Format where the voxels of interest have values 1
    """
    input_spec = ExtractMeanTSInputSpec
    output_spec = ExtractMeanTSOutputSpec

    def _run_interface(self, runtime):
                
        print 'in select_ts_with_mask'
        
        file_4D = self.inputs.file_4D
        mask_file = self.inputs.mask_file
        
        
        if isdefined(self.inputs.suffix):
            suffix = self.inputs.suffix
        else:
            suffix = "suf"
            
            
        print "loading img data " + file_4D

        ### Reading 4D volume file to extract time series
        img = nib.load(file_4D)
        img_data = img.get_data()
        
        print img_data.shape

        
        print "loading mask data " + mask_file

        ### Reading 4D volume file to extract time series
        mask_data = nib.load(mask_file).get_data()
        
        print mask_data.shape

        print "mean_select_mask_data"
        
        ### Retaining only time series who are within the mask + non_zero
        mean_masked_ts = mean_select_mask_data(img_data,mask_data)
        
        print "saving mean_masked_ts"
        mean_masked_ts_file = os.path.abspath('mean_' + suffix + '_ts.txt')    
        np.savetxt(mean_masked_ts_file,mean_masked_ts,fmt = '%.3f')
        
        
        print "plotting mean_masked_ts"
        
        plot_mean_masked_ts_file = os.path.abspath('mean_' + suffix + '_ts.eps')    
        
        plot_signals(plot_mean_masked_ts_file,mean_masked_ts)
        
        return runtime
        
        #return mean_masked_ts_file,subj_coord_rois_file
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        if isdefined(self.inputs.suffix):
        
            suffix = self.inputs.suffix
        
        else:
            
            suffix = "suf"
            
        outputs["mean_masked_ts_file"] = os.path.abspath('mean_' + suffix + '_ts.txt')
    
        return outputs
################################################################################# RegressCovar ######################################################################################################################
 

#from dmgraphanalysis_nodes.utils_cor import regress_movement_wm_csf_parameters
from dmgraphanalysis_nodes.utils_cor import regress_parameters,regress_filter_normalize_parameters

class RegressCovarInputSpec(BaseInterfaceInputSpec):
    masked_ts_file = File(exists=True, desc='time series in npy format', mandatory=True)
    
    rp_file = File(exists=True, desc='Movement parameters', mandatory=True)
    
    mean_wm_ts_file = File(exists=True, desc='White matter signal', mandatory=False)
    
    mean_csf_ts_file = File(exists=True, desc='Cerebro-spinal fluid (ventricules) signal', mandatory=False)
    
    filtered = traits.Bool(True, usedefault = True , desc = "Is the signal filtered after regression?")
    
    normalized = traits.Bool(True, usedefault = True , desc = "Is the signal normalized after regression?")
    
class RegressCovarOutputSpec(TraitedSpec):
    
    resid_ts_file = File(exists=True, desc="residuals of time series after regression of all paramters")
    

class RegressCovar(BaseInterface):
    """
    Regress parameters of non-interest (i.e. movement parameters, white matter, csf) from signal
    Optionnally filter and normalize (z-score) the residuals
    """
    input_spec = RegressCovarInputSpec
    output_spec = RegressCovarOutputSpec

    def _run_interface(self, runtime):
                    
                    
        print "in regress_covariates"
        
        masked_ts_file = self.inputs.masked_ts_file
        rp_file = self.inputs.rp_file
        filtered = self.inputs.filtered
        normalized = self.inputs.normalized
        
        print "load masked_ts_file"
        
        data_mask_matrix = np.loadtxt(masked_ts_file)
        
        print data_mask_matrix.shape

        print "load rp parameters"
        
        print rp_file
        
        rp = np.genfromtxt(rp_file)
        #rp = np.loadtxt(rp_file,dtype = np.float)
        
        print rp.shape
        
        if isdefined(self.inputs.mean_csf_ts_file):
            
            mean_csf_ts_file = self.inputs.mean_csf_ts_file
            
            print "load mean_csf_ts_file" + str(mean_csf_ts_file)
            
            mean_csf_ts = np.loadtxt(mean_csf_ts_file)
            
            print mean_csf_ts.shape
            
            #rp = np.concatenate((rp,mean_csf_ts),axis = 1)
            rp = np.concatenate((rp,mean_csf_ts.reshape(mean_csf_ts.shape[0],1)),axis = 1)
            
            print rp.shape
            
            
        if isdefined(self.inputs.mean_wm_ts_file):
            
            mean_wm_ts_file = self.inputs.mean_wm_ts_file
            
            print "load mean_wm_ts_file"
            
            mean_wm_ts = np.loadtxt(mean_wm_ts_file)
            
            
            #rp = np.concatenate((rp,mean_csf_ts),axis = 1)
            rp = np.concatenate((rp,mean_wm_ts.reshape(mean_wm_ts.shape[0],1)),axis = 1)
            
            print rp.shape
            
        
        if filtered == True and normalized == True:
            
            ### regression movement parameters and computing z-score on the residuals
            #resid_data_matrix = regress_movement_wm_csf_parameters(data_mask_matrix,rp,mean_wm_ts,mean_csf_ts)
            resid_data_matrix,resid_filt_data_matrix,z_score_data_matrix = regress_filter_normalize_parameters(data_mask_matrix,rp)
            
            print resid_data_matrix.shape

            print "saving resid_ts"
            
            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file,z_score_data_matrix )

            print "plotting resid_ts"
            
            plot_resid_ts_file = os.path.abspath('resid_ts.eps')
            
            plot_sep_signals(plot_resid_ts_file,z_score_data_matrix)
            
            
            print "plotting diff filtered and non filtered data"
            
            plot_diff_filt_ts_file = os.path.abspath('diff_filt_ts.eps')
            
            plot_signals(plot_diff_filt_ts_file,np.array(resid_filt_data_matrix - resid_data_matrix,dtype = 'float'))
            
        elif filtered == False and normalized == False:
            
            print "Using only regression"
        
            ### regression movement parameters and computing z-score on the residuals
            #resid_data_matrix = regress_movement_wm_csf_parameters(data_mask_matrix,rp,mean_wm_ts,mean_csf_ts)
            resid_data_matrix = regress_parameters(data_mask_matrix,rp)
            
            print resid_data_matrix.shape

            print "saving resid_ts"
            
            resid_ts_file = os.path.abspath('resid_ts.npy')
            np.save(resid_ts_file,resid_data_matrix )

            
            resid_ts_txt_file = os.path.abspath('resid_ts.txt')
            np.savetxt(resid_ts_txt_file,resid_data_matrix,fmt = '%0.3f')

            
            print "plotting resid_ts"
            
            plot_resid_ts_file = os.path.abspath('resid_ts.eps')
            
            plot_sep_signals(plot_resid_ts_file,resid_data_matrix)
            
            
            #print "plotting diff filtered and non filtered data"
            
            #plot_diff_filt_ts_file = os.path.abspath('diff_filt_ts.eps')
            
            #plot_signals(plot_diff_filt_ts_file,np.array(resid_filt_data_matrix - resid_data_matrix,dtype = 'float'))
            
        else:
            
            print "Warning, not implemented (RegressCovar)"
            
        return runtime
        
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["resid_ts_file"] = os.path.abspath('resid_ts.npy')
    
        return outputs

        ################################################################################# FindSPMRegressor ######################################################################################################################
 

class FindSPMRegressorInputSpec(BaseInterfaceInputSpec):
    
    spm_mat_file = File(exists=True, desc='SPM design matrix after generate model', mandatory=True)
    
    regressor_name = traits.String(exists=True, desc='Name of the regressor in SPM design matrix to be looked after', mandatory=True)
    
    run_index = traits.Int(1, usedefault = True , desc = "Run (session) index, default is one in SPM")
    
    only_positive_values = traits.Bool(True, usedefault = True , desc = "Return only positive values of the regressor (negative values are set to 0)")
    
class FindSPMRegressorOutputSpec(TraitedSpec):
    
    regressor_file = File(exists=True, desc="txt file containing the regressor")
    
class FindSPMRegressor(BaseInterface):
    """
    Regress parameters of non-interest (i.e. movement parameters, white matter, csf) from signal
    Optionnally filter and normalize (z-score) the residuals
    """
    input_spec = FindSPMRegressorInputSpec
    output_spec = FindSPMRegressorOutputSpec

    def _run_interface(self, runtime):
                   
                   
                    
        import scipy.io
        import numpy as np
        import os

        #print spm_mat_file
        
        
        spm_mat_file = self.inputs.spm_mat_file
        regressor_name = self.inputs.regressor_name
        run_index = self.inputs.run_index
        only_positive_values = self.inputs.only_positive_values
        
        
        print spm_mat_file
        
        ##Reading spm.mat for regressors extraction:
        d = scipy.io.loadmat(spm_mat_file)
        
        #print d
        
        
        ##Choosing the column according to the regressor name
        #_,col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == u'Sn(1) ' + regressor_name)
        
        cond_name = u'Sn(' + str(run_index) + ') ' + regressor_name+'*bf(1)'
        
        print cond_name
        
        _,col = np.where(d['SPM']['xX'][0][0]['name'][0][0] == cond_name)
        
        print col
        
        ## reformating matrix (1,len) in vector (len)
        regressor_vect = d['SPM']['xX'][0][0]['X'][0][0][:,col].reshape(-1)
        

        print regressor_vect
        
        if only_positive_values == True:
            
            regressor_vect[regressor_vect < 0] = 0
        
        print "Saving extract_cond"
        regressor_file = os.path.abspath('extract_cond.txt')

        np.savetxt(regressor_file,regressor_vect)

        return runtime
        
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["regressor_file"] = os.path.abspath('extract_cond.txt')
        
        return outputs

        
        ################################################################################# MergeRuns ######################################################################################################################
 

class MergeRunsInputSpec(BaseInterfaceInputSpec):
    
    ts_files = traits.List(File(exists=True), desc='Numpy files with time series from different runs (sessions)',mandatory=True)

    regressor_files = traits.List(File(exists=True), desc='Txt files with regressors from different runs (sessions)', mandatory=True)
    
    coord_rois_files = traits.List(File(exists=True), desc='Txt files with coords from different runs (sessions)', mandatory=True)
    
class MergeRunsOutputSpec(TraitedSpec):
    
    ts_all_runs_file = File(exists=True, desc="npy file containing the merge ts")
    
    regressor_all_runs_file = File(exists=True, desc="txt file containing the merged regressors")
    
    coord_rois_all_runs_file = File(exists=True, desc="txt file containing the merged coords")
    
class MergeRuns(BaseInterface):
    """
    Merge time series,regressor files and coord files
    Could be done with different cases
    """
    input_spec = MergeRunsInputSpec
    output_spec = MergeRunsOutputSpec

    def _run_interface(self, runtime):
                   
        print 'in merge_runs'
        
        ts_files = self.inputs.ts_files
        regressor_files = self.inputs.regressor_files
        coord_rois_files = self.inputs.coord_rois_files
        
        
        if len(ts_files) != len(regressor_files):
            
            print "Warning, time series and regressors have different length (!= number of runs)"
            return 0
            
        if len(ts_files) != len(coord_rois_files):
            
            print "Warning, time series and number of coordinates have different length (!= number of runs)"
            return 0
            
        ### concatenate time series
        for i,ts_file in enumerate(ts_files):
            
            data_matrix = np.load(ts_file)
            
            print data_matrix.shape
            
            ## loading ROI coordinates
            coord_rois = np.loadtxt(coord_rois_files[i])
            
            print coord_rois.shape
            
            if i == 0:
                data_matrix_all_runs = np.empty((data_matrix.shape[0],0),dtype = data_matrix.dtype)
                
                coord_rois_all_runs = np.array(coord_rois,dtype = 'float')
                
                
            if coord_rois_all_runs.shape[0] != coord_rois.shape[0]:
                
                print "ROIs do not match for all different sessions "
                
                print os.getcwd()
                
                print "Warning, not implemented yet.... "
                
                ### pris de http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
                ### à tester....
                ### finir également la partie avec data_matrix_all_runs, en supprimant les colonnes qui ne sont pas communes à tous les runs...
                
                0/0
                
                A = coord_rois_all_runs
                B = coord_rois
                
                nrows, ncols = A.shape
                dtype={'names':['f{}'.format(i) for i in range(ncols)],
                    'formats':ncols * [A.dtype]}

                C = np.intersect1d(A.view(dtype), B.view(dtype))

                # This last bit is optional if you're okay with "C" being a structured array...
                C = C.view(A.dtype).reshape(-1, ncols)

                coord_rois_all_runs = C

            data_matrix_all_runs = np.concatenate((data_matrix_all_runs,data_matrix),axis = 1)
            
            print data_matrix_all_runs.shape
            
        ### save times series for all runs
        ts_all_runs_file = os.path.abspath('ts_all_runs.npy')
        
        np.save(ts_all_runs_file,data_matrix_all_runs)
        
        ### save coords in common for all runs
        coord_rois_all_runs_file = os.path.abspath('coord_rois_all_runs.txt')
        
        np.savetxt(coord_rois_all_runs_file,coord_rois_all_runs, fmt = '%2.3f')
        
        ### compute regressor for all sessions together (need to sum)
        
        print "compute regressor for all sessions together (need to sum)"
        
        regressor_all_runs = np.empty(shape = (0), dtype = float)
        
        ### Sum regressors
        for i,regress_file in enumerate(regressor_files):
            
            regress_data_vector = np.loadtxt(regress_file)
            
            if regress_data_vector.shape[0] != 0:
                
                if regressor_all_runs.shape[0] == 0:
                    
                    regressor_all_runs = regress_data_vector
                else:
                    regressor_all_runs = regressor_all_runs + regress_data_vector
                
            print np.sum(regressor_all_runs != 0.0)
        
        regressor_all_runs_file = os.path.abspath('regressor_all_runs.txt')

        np.savetxt(regressor_all_runs_file,regressor_all_runs,fmt = '%0.3f')

        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        print outputs
        
        outputs["ts_all_runs_file"] = os.path.abspath('ts_all_runs.npy')
        
        outputs["coord_rois_all_runs_file"] = os.path.abspath('coord_rois_all_runs.txt')
        
        outputs["regressor_all_runs_file"] = os.path.abspath('regressor_all_runs.txt')

        return outputs

        ################################################################################# ComputeConfCorMat ######################################################################################################################
 
from dmgraphanalysis.utils_cor import return_conf_cor_mat

from dmgraphanalysis.utils_plot import plot_hist,plot_cormat
        
class ComputeConfCorMatInputSpec(BaseInterfaceInputSpec):
    
    ts_file = File(exists=True, desc='Numpy files with time series to be correlated',mandatory=True)

    weight_file = File(exists=True, desc='Weight of the correlation (normally, condition regressor file)', mandatory=False)
    
    conf_interval_prob = traits.Float(0.05, usedefault = True, desc='Confidence interval', mandatory=True)
    
    plot_mat = traits.Bool(True, usedefault = True, desc='Confidence interval', mandatory=False)
    
class ComputeConfCorMatOutputSpec(TraitedSpec):
    
    cor_mat_file = File(exists=True, desc="npy file containing the R values of correlation")
    
    Z_cor_mat_file = File(exists=True, desc="npy file containing the Z-values (after Fisher's R-to-Z trasformation) of correlation")
    
    conf_cor_mat_file = File(exists=True, desc="npy file containing the confidence interval around R values")
    
class ComputeConfCorMat(BaseInterface):
    """
    Compute correlation between time series, with a given confidence interval. If weight_file is specified, used for weighted correlation
    """
    input_spec = ComputeConfCorMatInputSpec
    output_spec = ComputeConfCorMatOutputSpec

    def _run_interface(self, runtime):
                   
                  
        print 'in compute_conf_correlation_matrix'
        
        ts_file = self.inputs.ts_file
        weight_file = self.inputs.weight_file
        conf_interval_prob = self.inputs.conf_interval_prob
            
        plot_mat = self.inputs.plot_mat
        
        
        print 'load resid data'
        
        data_matrix = np.load(ts_file)
        
        print data_matrix.shape
        
        print np.transpose(data_matrix).shape
        
        if isdefined(weight_file):
        
            print 'load weight_vect'
        
            weight_vect = np.loadtxt(weight_file)
            
            print weight_vect.shape
        
        else:
            weight_vect = np.ones(shape = (data_matrix.shape[1]))
        
        print "compute return_Z_cor_mat"
        
        cor_mat,Z_cor_mat,conf_cor_mat = return_conf_cor_mat(np.transpose(data_matrix),weight_vect,conf_interval_prob)
        
        print cor_mat.shape
        
        print Z_cor_mat.shape
        
        print conf_cor_mat.shape
        
        ### 
        print "saving cor_mat as npy"
        
        cor_mat_file = os.path.abspath('cor_mat.npy')
        
        np.save(cor_mat_file,cor_mat)
        
        print "saving conf_cor_mat as npy"
        
        conf_cor_mat_file = os.path.abspath('conf_cor_mat.npy')
        
        np.save(conf_cor_mat_file,conf_cor_mat)
        
        print "saving Z_cor_mat as npy"
        
        Z_cor_mat_file = os.path.abspath('Z_cor_mat.npy')
        
        np.save(Z_cor_mat_file,Z_cor_mat)
        
        
        if plot_mat == True:
                
            ############# cor_mat
            
            #### heatmap 
            
            print 'plotting cor_mat heatmap'
            
            plot_heatmap_cor_mat_file =  os.path.abspath('heatmap_cor_mat.eps')
            
            plot_cormat(plot_heatmap_cor_mat_file,cor_mat,list_labels = [])
            
            #### histogram 
            
            print 'plotting cor_mat histogram'
            
            plot_hist_cor_mat_file = os.path.abspath('hist_cor_mat.eps')
            
            plot_hist(plot_hist_cor_mat_file,cor_mat,nb_bins = 100)
            
            ############ Z_cor_mat
            
            Z_cor_mat = np.load(Z_cor_mat_file)
            
            #### heatmap 
            
            print 'plotting Z_cor_mat heatmap'
            
            plot_heatmap_Z_cor_mat_file =  os.path.abspath('heatmap_Z_cor_mat.eps')
            
            plot_cormat(plot_heatmap_Z_cor_mat_file,Z_cor_mat,list_labels = [])
            
            #### histogram 
            
            print 'plotting Z_cor_mat histogram'
            
            plot_hist_Z_cor_mat_file = os.path.abspath('hist_Z_cor_mat.eps')
            
            plot_hist(plot_hist_Z_cor_mat_file,Z_cor_mat,nb_bins = 100)
            
            ############ conf_cor_mat
            
            #### heatmap 
            
            print 'plotting conf_cor_mat heatmap'
            
            plot_heatmap_conf_cor_mat_file =  os.path.abspath('heatmap_conf_cor_mat.eps')
            
            plot_cormat(plot_heatmap_conf_cor_mat_file,conf_cor_mat,list_labels = [])
            
            #### histogram 
            
            print 'plotting conf_cor_mat histogram'
            
            plot_hist_conf_cor_mat_file = os.path.abspath('hist_conf_cor_mat.eps')

            plot_hist(plot_hist_conf_cor_mat_file,conf_cor_mat,nb_bins = 100)
            
        
        
        
        return runtime
        
    def _list_outputs(self):
        
        outputs = self._outputs().get()
        
        outputs["cor_mat_file"] = os.path.abspath('cor_mat.npy')
        
        outputs["conf_cor_mat_file"] = os.path.abspath('conf_cor_mat.npy')
        
        outputs["Z_cor_mat_file"] = os.path.abspath('Z_cor_mat.npy')
        
        print outputs
        
        return outputs

################################## pearson correl between reg and time series #######################################

#def correl_ts_with_reg(ts_file,regressor_file):

    #import numpy as np
    #import os
    
    #from scipy.stats import pearsonr
    #from scipy.stats.mstats import zscore
    
    
    #print 'load regressor_vect'
    
    #regressor_vect = np.loadtxt(regressor_file)
    
    #print regressor_vect.shape
    
    
    #print 'load ts data'
    
    #resid_data_matrix = np.load(ts_file)
    
    #print resid_data_matrix.shape
    
    #if regressor_vect.shape[0] == resid_data_matrix.shape[1]:
        
        #print "Regressor size compatible with ts length"
        
    #elif regressor_vect.shape[0] == resid_data_matrix.shape[0]:
        
        #print "Warning, should transpose ts matrix, but keeping computation"
        
        #resid_data_matrix = np.transpose(resid_data_matrix)
    
    #else:
        #print "Warning, regressor size %d incompatible with ts length %d %d, breaking"%(regressor_vect.shape[0],resid_data_matrix.shape[0],resid_data_matrix.shape[1])
        
        #sys.exit()
        
        
        
    #print resid_data_matrix[0,:10]
    
    ##zscore_data_matrix = zscore(resid_data_matrix,axis = 0)
    
    #zscore_data_matrix = np.zeros(shape = resid_data_matrix.shape,dtype = resid_data_matrix.dtype)
    
    #for i in range(zscore_data_matrix.shape[0]):
    
        #zscore_data_matrix[i,:] = zscore(resid_data_matrix[i,:])
    
    #print zscore_data_matrix[0,:10]
    
    #from utils_plot import plot_sep_signals
    
    #plot_zscore_data_matrix_file = os.path.abspath("Zscore_regressor.eps")
    
    #regressor_vect2 = regressor_vect.reshape(1,regressor_vect.shape[0])
    
    #print regressor_vect2.shape
    
    #plot_data = np.concatenate((regressor_vect2,zscore_data_matrix),axis = 0)
    
    #print plot_data.shape
    
    #plot_sep_signals(plot_zscore_data_matrix_file,plot_data)
    
    
    
    
    #correl_res = []
    
    #for i in range(zscore_data_matrix.shape[0]):
        
        #correl_res.append(list(pearsonr(zscore_data_matrix[i,:],regressor_vect)))
    
    #print np.array(correl_res)
    
        
        
        
        
    ##correl_res = []
    
    ##for i in range(resid_data_matrix.shape[0]):
        
        
    
        ##correl_res.append(list(pearsonr(resid_data_matrix[i,:],regressor_vect)))
    
    ##print np.array(correl_res)
    
    
    #correl_ROI_ts_file = os.path.abspath('correl_ts_with_reg.txt')

    #np.savetxt(correl_ROI_ts_file,np.array(correl_res),fmt = '%0.5f')
    
    
    
    
    #return correl_ROI_ts_file

    
#def compute_conf_correlation_matrix(resid_ts_file,regressor_file,conf_interval_prob):
    
    #import rpy,os
    #import nibabel as nib
    #import numpy as np
    
    #from dmgraphanalysis.utils_cor import return_conf_cor_mat
    
    
    #print 'load regressor_vect'
    
    #regressor_vect = np.loadtxt(regressor_file)
    
    
    #print 'load resid data'
    
    #resid_data_matrix = np.load(resid_ts_file)
    
    #print "compute return_Z_cor_mat"
    
    #print resid_data_matrix.shape
    #print np.transpose(resid_data_matrix).shape
    
    #cor_mat,Z_cor_mat,conf_cor_mat = return_conf_cor_mat(np.transpose(resid_data_matrix),regressor_vect,conf_interval_prob)
    
    #print cor_mat.shape
    
    #print Z_cor_mat.shape
    
    #print conf_cor_mat.shape
    
    #### 
    #print "saving cor_mat as npy"
    
    #cor_mat_file = os.path.abspath('cor_mat.npy')
    
    #np.save(cor_mat_file,cor_mat)
    
    #print "saving conf_cor_mat as npy"
    
    #conf_cor_mat_file = os.path.abspath('conf_cor_mat.npy')
    
    #np.save(conf_cor_mat_file,conf_cor_mat)
    
    #print "saving Z_cor_mat as npy"
    
    #Z_cor_mat_file = os.path.abspath('Z_cor_mat.npy')
    
    #np.save(Z_cor_mat_file,Z_cor_mat)
    
    #return cor_mat_file,Z_cor_mat_file,conf_cor_mat_file

#def compute_Z_correlation_matrix(resid_ts_file,regressor_file):
    
    #import rpy,os
    #import nibabel as nib
    #import numpy as np
    
    #from dmgraphanalysis.utils_cor import return_Z_cor_mat
    
    
    #print 'load regressor_vect'
    
    #regressor_vect = np.loadtxt(regressor_file)
    
    
    #print 'load resid data'
    
    #resid_data_matrix = np.load(resid_ts_file)
    
    #print "compute return_Z_cor_mat"
    
    #print resid_data_matrix.shape
    #print np.transpose(resid_data_matrix).shape
    
    #Z_cor_mat = return_Z_cor_mat(np.transpose(resid_data_matrix),regressor_vect)
    
    #print Z_cor_mat.shape
    
    #### 
    #print "saving Z_cor_mat as npy"
    
    #Z_cor_mat_file = os.path.abspath('Z_cor_mat.npy')
    
    #np.save(Z_cor_mat_file,Z_cor_mat)
    
    #return Z_cor_mat_file

    
#def plot_hist_var_cor_mat(cor_mat_file,sderr_cor_mat_file,pval_cor_mat_file):

    #import os
    #import numpy as np
    
    #from dmgraphanalysis.utils_plot import plot_hist,plot_cormat
    
    #import nibabel as nib
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    ############ cor_mat
    
    #cor_mat = np.load(cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting cor_mat heatmap'
    
    #plot_heatmap_cor_mat_file =  os.path.abspath('heatmap_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_cor_mat_file,cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting cor_mat histogram'
    
    #plot_hist_cor_mat_file = os.path.abspath('hist_cor_mat.eps')
    
    #plot_hist(plot_hist_cor_mat_file,cor_mat,nb_bins = 100)
    
    
    ########### sderr_cor_mat 
    
    #sderr_cor_mat = np.load(sderr_cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting sderr_cor_mat heatmap'
    
    #plot_heatmap_sderr_cor_mat_file =  os.path.abspath('heatmap_sderr_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_sderr_cor_mat_file,sderr_cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting sderr_cor_mat histogram'
    
    #plot_hist_sderr_cor_mat_file = os.path.abspath('hist_sderr_cor_mat.eps')
    
    #plot_hist(plot_hist_sderr_cor_mat_file,sderr_cor_mat,nb_bins = 100)
    
    ############## pval cor_mat
    
    #pval_cor_mat = np.load(pval_cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting pval_cor_mat heatmap'
    
    #plot_heatmap_pval_cor_mat_file =  os.path.abspath('heatmap_pval_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_pval_cor_mat_file,pval_cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting pval_cor_mat histogram'
    
    #plot_hist_pval_cor_mat_file = os.path.abspath('hist_pval_cor_mat.eps')
    
    #plot_hist(plot_hist_pval_cor_mat_file,pval_cor_mat,nb_bins = 100)
    
    #return plot_hist_cor_mat_file,plot_heatmap_cor_mat_file,plot_hist_sderr_cor_mat_file,plot_heatmap_sderr_cor_mat_file,plot_hist_pval_cor_mat_file,plot_heatmap_pval_cor_mat_file
    
    
#def plot_hist_conf_cor_mat(cor_mat_file,Z_cor_mat_file,conf_cor_mat_file):

    #import os
    #import numpy as np
    
    #import nibabel as nib
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.utils_plot import plot_hist,plot_cormat
    
    ############# cor_mat
    
    #cor_mat = np.load(cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting cor_mat heatmap'
    
    #plot_heatmap_cor_mat_file =  os.path.abspath('heatmap_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_cor_mat_file,cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting cor_mat histogram'
    
    #plot_hist_cor_mat_file = os.path.abspath('hist_cor_mat.eps')
    
    #plot_hist(plot_hist_cor_mat_file,cor_mat,nb_bins = 100)
    
    ############# Z_cor_mat
    
    #Z_cor_mat = np.load(Z_cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting Z_cor_mat heatmap'
    
    #plot_heatmap_Z_cor_mat_file =  os.path.abspath('heatmap_Z_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_Z_cor_mat_file,Z_cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting Z_cor_mat histogram'
    
    #plot_hist_Z_cor_mat_file = os.path.abspath('hist_Z_cor_mat.eps')
    
    #plot_hist(plot_hist_Z_cor_mat_file,Z_cor_mat,nb_bins = 100)
    
    ############# conf_cor_mat
    
    #conf_cor_mat = np.load(conf_cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting conf_cor_mat heatmap'
    
    #plot_heatmap_conf_cor_mat_file =  os.path.abspath('heatmap_conf_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_conf_cor_mat_file,conf_cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting conf_cor_mat histogram'
    
    #plot_hist_conf_cor_mat_file = os.path.abspath('hist_conf_cor_mat.eps')

    #plot_hist(plot_hist_conf_cor_mat_file,conf_cor_mat,nb_bins = 100)
    
    #return plot_hist_cor_mat_file,plot_heatmap_cor_mat_file,plot_hist_Z_cor_mat_file,plot_heatmap_Z_cor_mat_file,plot_hist_conf_cor_mat_file,plot_heatmap_conf_cor_mat_file
    
#def plot_hist_Z_cor_mat(Z_cor_mat_file):

    #import os
    #import numpy as np
    
    #import nibabel as nib
    
    #from nipype.utils.filemanip import split_filename as split_f
    
    #from dmgraphanalysis.utils_plot import plot_hist,plot_cormat
    
    ########### Z_cor_mat
    
    #Z_cor_mat = np.load(Z_cor_mat_file)
    
    ##### heatmap 
    
    #print 'plotting Z_cor_mat heatmap'
    
    #plot_heatmap_Z_cor_mat_file =  os.path.abspath('heatmap_Z_cor_mat.eps')
    
    #plot_cormat(plot_heatmap_Z_cor_mat_file,Z_cor_mat,list_labels = [])
    
    ##### histogram 
    
    #print 'plotting Z_cor_mat histogram'
    
    #plot_hist_Z_cor_mat_file = os.path.abspath('hist_Z_cor_mat.eps')
    
    #plot_hist(plot_hist_Z_cor_mat_file,Z_cor_mat,nb_bins = 100)
    
    #return plot_hist_Z_cor_mat_file,plot_heatmap_Z_cor_mat_file
    
###### spm_mask and all infos from contrast index
#def extract_signif_contrast_mask(spm_contrast_index):

    #import os
    #from define_variables import nipype_analyses_path,peak_activation_mask_analysis_name,ROI_mask_prefix
    
    ##### indexed_mask
    #indexed_mask_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "indexed_mask-" + ROI_mask_prefix + "_spm_contrast" + str(spm_contrast_index) + ".nii")
        
    ##### saving ROI coords as textfile
    #### ijk coords
    #coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-" + ROI_mask_prefix + "_spm_contrast" + str(spm_contrast_index) + ".txt")

    #### coords in MNI space
    #MNI_coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-MNI-" + ROI_mask_prefix + "_spm_contrast" + str(spm_contrast_index) + ".txt")

    ##### saving ROI coords as textfile
    #label_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + "_spm_contrast" + str(spm_contrast_index) + ".txt")
    ##label_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + "_jane.txt")
        
    ##### all info in a text file
    #info_rois_file  =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "info-" + ROI_mask_prefix + "_spm_contrast" + str(spm_contrast_index) + ".txt")

    #return indexed_mask_rois_file,coord_rois_file,MNI_coord_rois_file,label_rois_file,info_rois_file
