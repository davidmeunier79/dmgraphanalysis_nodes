# -*- coding: utf-8 -*-
"""
Define main variables and imports 
to be set for before each run.



"""

#### basic imports
import sys,io,os,fnmatch,shutil

import matplotlib
matplotlib.use('PS')

#### nibabel import
import nibabel as nib

##### nipype import
#from nipype import config
#config.enable_debug_mode()
import nipype
print "Nipype version: " + nipype.__version__
print "Nipype path: " + nipype.__file__

import nipype.interfaces.io as nio #data grabber
import nipype.interfaces.spm as spm
import nipype.interfaces.dcm2nii as dn
import nipype.interfaces.freesurfer as fs    # freesurfer

import nipype.interfaces.matlab as mlab
#~ mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash") #comment on lance matlab sans la console matlab 

import nipype.interfaces.fsl as fsl
fsl.FSLCommand.set_default_output_type('NIFTI')

#import nipype.interfaces.fsl.BET
import nipype.algorithms.rapidart as ra #correction d'artefacts, volumes avec trop de mouvements, pbs dans le masque
import nipype.algorithms.modelgen as model 
import nipype.pipeline.engine as pe
from nipype.pipeline.engine import Workflow
#from nipype.interfaces.base import Bunch
from nipype.interfaces.utility import IdentityInterface, Function, Select, Rename


import nipype.interfaces.nipy.preprocess as nipy # interface, permet de traiter les .nifti en 4D comme un tableau np.array en 4D et après on peut faire toutes les analyses numpy

#from pandas import DataFrame

#from pandas import DataFrame

import numpy as np

import re

#import rpy
#from scipy import stats

import getpass

#### general path
if getpass.getuser() == 'sgarcia' and  sys.platform.startswith('linux'):
    #main_path = "/home/sgarcia/mnt/NEURO011/IRMf/14_fMRI_AVERSION_PNRA_2008/RESULTS/FONCTIONAL_DATA/PREPROCESSING/"
    main_path = os.path.abspath('../datasets/')+'/'
elif getpass.getuser() == 'david' and  sys.platform.startswith('linux'):
    
    ## sur le serveur
    #main_path = "/home/david/Mount/NEURO011/IRMf/18_EPISODIC/"
    
    ## sur le nouveau dserveur 
    
    main_path =  "/home/david/Mount/crnldata/crnldata/cmo/homme/IRMf/18_EPISODIC"
    
    ## direct sur le disque d'AL
    #main_path = "/media/Iomega_HDD/18_EPISODIC"
    
    # essai en local
    main_path_loc = "/media/speeddata/Data/Nipype-Episodic"
        
    ### ajout spm dans matlab "à la main"...
    matlab_spm_path='/home/david/matlab/spm8'
    mlab.MatlabCommand.set_default_paths(matlab_spm_path)
    
elif getpass.getuser() == 'alsaive' and  sys.platform.startswith('linux'):
    
    main_path = "/media/data/EPISODIC_IRMf/"
    
     ### ajout spm dans matlab "à la main"...
    matlab_spm_path='/home/alsaive/matlab/SPM8/'
    mlab.MatlabCommand.set_default_paths(matlab_spm_path)
    


# a changer et mettre chemin absolu
filename_hdf5 = os.path.abspath('../all_data.h5')


### ajout spm dans matlab "à la main"...
mlab.MatlabCommand.set_default_paths(matlab_spm_path)

#from nipype.interfaces.matlab import MatlabCommand
#res = MatlabCommand(script='''which(spm)''', paths=[matlab_spm_path], mfile=False).run()
#print res.runtime.stdout

print "Main path: " + main_path

nipype_analyses_path = main_path_loc + "/nipype_analyses/"

nifti_path = nipype_analyses_path + 'NiftiFiles_FS'

#ua_path = nipype_analyses_path + 'Files_ua_mod_hdr'
ua_path = nipype_analyses_path + 'Files_ua'

#uaf_path = nipype_analyses_path + 'Files_uaf'
uaf_path = nipype_analyses_path + 'Files_uaf_remove_last_scans'

#change_name_nifti_path = os.path.join(main_path,'C_CHANGE_NAME_NIFTI_DATA')

############################################################# loop all set of data

## run_correl_mat, semble qu'il y a un pb avec S26 (rp_ plus long d'une valeur qu'attendu...)
## corrigé en supprimant directement la derniere ligne dans Files_ua/FunctionalRuns_ua/rp_S26_Run*.txt
#subject_nums =['S02','S04','S05','S07','S08','S09','S10','S12','S13','S14','S15','S16','S18','S20','S22','S24','S25']

subject_nums =['S02','S04','S05','S07','S08','S09','S10','S12','S13','S14','S15','S16','S18','S20','S22','S24','S25','S26']

funct_run_indexs = [1, 2, 3]


######################################## Preprocessing parameters ########################################

## preprocessing analysis name

## with preprocessing unwarp_fsl
## correct normalisation is in smooth_modif
## 4th Whole Brain EPI is in smooth_forth_WBEPI

## !!!! nom utilisé pour le pretraitement précedent avant refonte des noms,
#preproc_name = "norm_funct_unwarp_fsl_realign_spm"


### with original ua files
#preproc_name = "norm_funct-ua_orig"
#preproc_name = "norm_funct-ua_orig-new_st_wbepi"

preproc_name = "norm_funct-ua_orig-new_st_wbepi-sans_image1-anat"
#preproc_name = "norm_funct-swua_orig_sans_image1"

### with original swua files
#preproc_name = "swua_orig"
#preproc_name = "swua_orig_remove_last_scans"

split_preproc_name = preproc_name.split('-')

## homw many scans are removed
nb_scans_to_remove = 1 
TR = 2.5

TR_WBEPI = 4.0
######################################## Contrasts definition #############################################

#l1_analysis_name = 'l1_analysis-Hit_model8-ua_orig-modif_norm' 

#l1_analysis_name = 'l1_analysis-WWW_What_model8-ua_orig-modif_norm-by_runs' 
#l1_analysis_name = 'l1_analysis-WWW_What_model8-ua_orig-modif_norm-no_concat_runs' 

#l1_analysis_name = 'l1_analysis-WWW_What_model8-swua_orig' 

### modified in last versions; now is Model8_Connectivity
#model_name = 'WWW_What_model8'


#model_name = 'Hit_model8'
#model_name = 'Model8_Memory_Hit'
#model_name = 'Model8_Connectivity'

#model_name = 'Model9_3blocs'
#model_name = 'Model9_Only3blocs_WWW_What'
#model_name = 'Model9_Only3blocs'

#model_name = 'Model9_Only3events'
#model_name = 'Model9_Only3events_Hit'
#model_name = 'Model9_Only3events_WWW_What'
model_name = 'Model9_Only3events_WWW'

split_model_name = model_name.split('_')

print split_model_name
#l1_analysis_name = 'l1_analysis_WWW_What_model8-unwarp_fsl-no_concat_runs' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-regress_mvt-respi' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-regress_mvt' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs' 

#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-sansPrepOdor' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-laggedrespi' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-masking'

l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-anticip'

### FIR, does not work so far
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-anticip-all_FIR' 
#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-anticip-each_FIR' 

#l1_analysis_name = 'l1_analysis-' + model_name + '-' + preproc_name + '-no_concat_runs-anticip_prep' 

split_l1_analysis_name = l1_analysis_name.split('-')

print split_l1_analysis_name

if 'all_FIR' in split_l1_analysis_name or 'each_FIR' in split_l1_analysis_name:
    
    fir_length = 10
    fir_order = 4
    
    #fir_length = 3
    #fir_order = 1

#### level_one SPM contrast
### regressors WWW_What
### contrasts (used in contrast_generator and condition_generator)

## obsolete in last versions
#if model_name == 'WWW_What_model8':
    ##contrasts_episodic_names = ['T +1 Hit-WWW','T +1 Hit-What'] # previous version (corresponds to Odor_Hit-WWW Odor_Hit-What in new version)
    #contrasts_episodic_names = ['T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Rec_Hit-WWW','T +1 Rec_Hit-What','T +1 Recall_Hit-WWW','T +1 Recall_Hit-What']
    
if model_name == 'Hit_model8':
    contrasts_episodic_names = ['T +1 Rec_hit','T +1 memory_of_What-S3','T +1 memory_of_WWW-S3']
    
    #contrasts_episodic_names = ['T +1 Rec_hit','T +1 memory_of_What-S3','T +1 memory_of_WWW-S3','T +1 memory_of_WWhere-S3','T +1 memory_of_WWhich-S3']
    #contrasts_episodic_names = ['T +1 Rec_hit']
    
    #contrasts_episodic_names = ['T +1 Rec_hit','T +1 Rec_miss','T +1 Rec_cr','T +1 Rec_fa','T +1 memory_of_WWW-S3 -1 memory_of_What-S3','T +1 memory_of_What-S3 -1 memory_of_WWW-S3']
    
elif model_name == 'Model8_Memory_Hit':
    contrasts_episodic_names = ['T +1 memory_Hit']
    
elif model_name == 'Model8_Connectivity':
    contrasts_episodic_names = ['T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Rec_Hit-WWW','T +1 Rec_Hit-What','T +1 Recall_Hit-WWW','T +1 Recall_Hit-What']
    

elif model_name == 'Model9_3blocs':
    
    contrasts_episodic_names = ['T +1 Odor_Hit-WWW -1 Odor_Hit-What','T +1 Odor_Hit-What -1 Odor_Hit-WWW','T +1 Odor_Hit-WWW +1 Block_Rec_Hit-WWW -1 Odor_Hit-What -1 Block_Rec_Hit-What','T -1 Odor_Hit-WWW -1 Block_Rec_Hit-WWW +1 Odor_Hit-What +1 Block_Rec_Hit-What']
    
    #contrasts_episodic_names = ['T +1 Odor_Hit-WWW -1 Odor_Hit-What','T +1 Odor_Hit-What -1 Odor_Hit-WWW']
    
    #contrasts_episodic_names = ['T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Rec_Hit-WWW','T +1 Rec_Hit-What','T +1 Recall_Hit-WWW','T +1 Recall_Hit-What','T +1 Block_Rec_Hit-WWW','T +1 Block_Rec_Hit-What','T +1 Block_Recall_Hit-WWW','T +1 Block_Recall_Hit-What','T +1 Elaboration_Hit-WWW','T +1 Elaboration_Hit-What']
    
elif "Model9" in split_model_name and 'Only3blocs' in split_model_name :
    #contrasts_episodic_names = ['T +1 Block_Rec_Hit-WWW','T +1 Block_Rec_Hit-What','T +1 Block_Rec_Hit-WWW -1 Block_Rec_Hit-What']
    
    if 'anticip_prep' in split_l1_analysis_name:
        contrasts_episodic_names = ['T +1 Odor_anticip_prep','T +1 Odor_anticip_prep -1 Rest']
        
    elif 'anticip' in split_l1_analysis_name:
        
        
        if "What" in split_model_name and "WWW" in split_model_name:
            
            print "Found WWW_What"
            #contrasts_episodic_names = ['T +1 Rest_Hit-What','T +1 Odor_anticipation_Hit-What','T +1 Odor_preparation_Hit-What','T +1 Odor_Hit-What','T +1 Rec_Hit-What','T +1 Recall_Hit-What','T +1 Rest_Hit-What','T +1 Odor_anticipation_Hit-What','T +1 Odor_preparation_Hit-What','T +1 Odor_Hit-What','T +1 Rec_Hit-What','T +1 Recall_Hit-What','T +1 Rest_Hit-What -1 Rest_Hit-What',]
            contrasts_episodic_names = ['T +1 Rest_Hit-WWW','T +1 Block_Odor_anticipation_Hit-WWW','T +1 Block_Odor_preparation_Hit-WWW','T +1 Block_Rec_Hit-WWW','T +1 Block_Recall_Hit-WWW','T +1 Elaboration_Hit-WWW','T +1 Rest_Hit-What','T +1 Block_Odor_anticipation_Hit-What','T +1 Block_Odor_preparation_Hit-What','T +1 Block_Rec_Hit-What','T +1 Block_Recall_Hit-What','T +1 Elaboration_Hit-What']
            #contrasts_episodic_names = ['T +1 Rest_Hit-WWW -1 Rest_Hit-What','T +1 Block_Odor_anticipation_Hit-WWW -1 Block_Odor_anticipation_Hit-What','T +1 Block_Odor_preparation_Hit-WWW -1 Block_Odor_preparation_Hit-What','T +1 Block_Rec_Hit-WWW -1 Block_Rec_Hit-What','T +1 Block_Recall_Hit-WWW -1 Block_Recall_Hit-What','T +1 Elaboration_Hit-WWW -1 Elaboration_Hit-What']
            
        else:
        
            contrasts_episodic_names = ['T +1 Rest','T +1 Block_Odor_anticipation','T +1 Block_Odor_preparation','T +1 Block_Rec','T +1 Block_Recall','T +1 Elaboration']
 
elif "Model9" in split_model_name and 'Only3events' in split_model_name :
    #contrasts_episodic_names = ['T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Rec_Hit-WWW','T +1 Rec_Hit-What','T +1 Recall_Hit-WWW','T +1 Recall_Hit-What']
    #contrasts_episodic_names = ['T +1 Odor_preparation','T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Odor_Hit-WWW -1 Odor_Hit-What']
    
    #contrasts_episodic_names = ['T +1 Odor_preparation','T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Odor_Hit-WWW -1 Odor_Hit-What','T +1 Odor_Hit-WWW -1 Odor_preparation','T +1 Odor_Hit-What -1 Odor_preparation']
    
    
    #contrasts_episodic_names = ['T +1 Preparation-WWW','T +1 Preparation-What','T +1 Odor_Hit-WWW','T +1 Odor_Hit-What','T +1 Preparation-WWW -1 Preparation-What']

    #
    if 'anticip_prep' in split_l1_analysis_name:
        contrasts_episodic_names = ['T +1 Odor_anticip_prep','T +1 Odor_anticip_prep -1 Rest']
        
    elif 'anticip' in split_l1_analysis_name:
        
        print "Found anticip"
        
        if "WWW" in split_model_name and "What" in split_model_name:
            
            print "Found WWW_What"
            #contrasts_episodic_names = ['T +1 Rest_Hit-WWW','T +1 Odor_anticipation_Hit-WWW','T +1 Odor_preparation_Hit-WWW','T +1 Odor_Hit-WWW','T +1 Rec_Hit-WWW','T +1 Recall_Hit-WWW','T +1 Rest_Hit-What','T +1 Odor_anticipation_Hit-What','T +1 Odor_preparation_Hit-What','T +1 Odor_Hit-What','T +1 Rec_Hit-What','T +1 Recall_Hit-What','T +1 Rest_Hit-WWW -1 Rest_Hit-What',]
            #contrasts_episodic_names = ['T +1 Rest_Hit-WWW','T +1 Odor_anticipation_Hit-WWW','T +1 Odor_preparation_Hit-WWW','T +1 Odor_Hit-WWW','T +1 Rec_Hit-WWW','T +1 Recall_Hit-WWW','T +1 Rest_Hit-What','T +1 Odor_anticipation_Hit-What','T +1 Odor_preparation_Hit-What','T +1 Odor_Hit-What','T +1 Rec_Hit-What','T +1 Recall_Hit-What']
            
            
            contrasts_episodic_names = [                'T +1 Odor_Hit-WWW -1 Odor_Hit-What',
                'T +1 Rec_Hit-WWW -1 Rec_Hit-What',
                'T +1 Recall_Hit-WWW -1 Recall_Hit-What',
                'T -1 Odor_Hit-WWW +1 Odor_Hit-What',
                'T -1 Rec_Hit-WWW +1 Rec_Hit-What',
                'T -1 Recall_Hit-WWW +1 Recall_Hit-What']
                
            
            #contrasts_episodic_names = [                'T +1 Rest_Hit-WWW -1 Rest_Hit-What',
                #'T +1 Odor_anticipation_Hit-WWW -1 Odor_anticipation_Hit-What',
                ##'T +1 Odor_preparation_Hit-WWW -1 Odor_preparation_Hit-What',
                #'T +1 Odor_Hit-WWW -1 Odor_Hit-What',
                #'T +1 Rec_Hit-WWW -1 Rec_Hit-What',
                #'T +1 Recall_Hit-WWW -1 Recall_Hit-What',
                #'T -1 Rest_Hit-WWW +1 Rest_Hit-What',
                #'T -1 Odor_anticipation_Hit-WWW +1 Odor_anticipation_Hit-What',
                ##'T -1 Odor_preparation_Hit-WWW +1 Odor_preparation_Hit-What',
                #'T -1 Odor_Hit-WWW +1 Odor_Hit-What',
                #'T -1 Rec_Hit-WWW +1 Rec_Hit-What',
                #'T -1 Recall_Hit-WWW +1 Recall_Hit-What']
                
            #contrasts_episodic_names = ['T +1 Rest_Hit-WWW -1 Rest_Hit-What',
                #'T +1 Odor_anticipation_Hit-WWW -1 Odor_anticipation_Hit-What',
                #'T +1 Odor_preparation_Hit-WWW -1 Odor_preparation_Hit-What',
                #'T +1 Odor_Hit-WWW -1 Odor_Hit-What',
                #'T +1 Rec_Hit-WWW -1 Rec_Hit-What',
                #'T +1 Recall_Hit-WWW -1 Recall_Hit-What',
                #'T -1 Rest_Hit-WWW +1 Rest_Hit-What',
                #'T -1 Odor_anticipation_Hit-WWW +1 Odor_anticipation_Hit-What',
                #'T -1 Odor_preparation_Hit-WWW +1 Odor_preparation_Hit-What',
                #'T -1 Odor_Hit-WWW +1 Odor_Hit-What',
                #'T -1 Rec_Hit-WWW +1 Rec_Hit-What',
                #'T -1 Recall_Hit-WWW +1 Recall_Hit-What']
            
        elif "Hit" in split_model_name:
        
            #contrasts_episodic_names = ['T +1 Rest','T +1 Odor_anticipation','T +1 Odor_preparation','T +1 Odor','T +1 Rec','T +1 Recall']
            ### Model9 no prep
            contrasts_episodic_names = ['T +1 Rest_hit','T +1 Odor_anticipation_hit','T +1 Odor_hit','T +1 Rec_hit','T +1 Recall_hit']
            
        else:
        
            #contrasts_episodic_names = ['T +1 Rest','T +1 Odor_anticipation','T +1 Odor_preparation','T +1 Odor','T +1 Rec','T +1 Recall']
            ### Model9 no prep
            contrasts_episodic_names = ['T +1 Rest','T +1 Odor_anticipation','T +1 Odor','T +1 Rec','T +1 Recall']
            
            

numberOfContrasts = len(contrasts_episodic_names) #number of contrasts you specified in the first level analysis
contrast_indexes = range(1,numberOfContrasts+1)

#### level_two SPM contrast

#l2_analysis_name = 'l2_analysis-Hit_model8-OneSampleTTest-ua_orig-modif_norm'

#l2_analysis_name = 'l2_analysis-WWW_What_model8-OneSampleTTest-ua_orig-modif_norm'
#l2_analysis_name = 'l2_analysis-WWW_What_model8-OneSampleTTest-swua_orig'
#l2_analysis_name = 'l2_analysis-OneSampleTTest' + model_name + '-' + preproc_name + '-no_concat_runs'
#l2_analysis_name = 'l2_analysis-OneSampleTTest' + model_name + '-' + preproc_name + '-no_concat_runs-sansPrepOdor'
#l2_analysis_name = 'l2_analysis-OneSampleTTest' + model_name + '-' + preproc_name + '-no_concat_runs-avecPrepOdor'
l2_analysis_name = 'l2_analysis-OneSampleTTest' + model_name + '-' + preproc_name + '-no_concat_runs'

if 'respi' in split_l1_analysis_name:
    l2_analysis_name = l2_analysis_name + '-respi'
    
    
if 'laggedrespi' in split_l1_analysis_name:
    l2_analysis_name = l2_analysis_name + '-laggedrespi'
   
    nb_laggedrespi_TR = 8 ## de 0 (syncro) à n-1
    
if 'masking' in split_l1_analysis_name:
    l2_analysis_name = l2_analysis_name + '-masking'
    
if 'anticip' in split_l1_analysis_name:
    l2_analysis_name = l2_analysis_name + '-anticip'
    
if 'anticip_prep' in split_l1_analysis_name:
    l2_analysis_name = l2_analysis_name + '-anticip_prep'
    
#l2_analysis_name = 'l2_analysis-_WWW_What_model8-OneSampleTTest-unwarp_fsl-no_concat_runs'

cont_group1 = ('Group_1','T', ['mean'],[1])

group_contrasts = [[cont_group1]]
numberOfGroupContrasts = len(group_contrasts)

print "Number of contrasts (One Sample TTest): " + str(numberOfGroupContrasts)

group_contrast_indexes = range(0,numberOfGroupContrasts)

######################################### correlation with behav results #####################################

## older version of the score
scores_epi = [0.78, -1.30, 2.39, -0.91, -0.72, 1.05, 0.22, 1.81, -1.17, -0.10, -1.03, -0.49, 0.12, -1.24, 1.18, -1.05, -0.29, 0.69]
print len(scores_epi)

# new version (score total)
scores_total = [3.33,-3.20,6.59,-2.07,-1.32,1.85,1.13,7.23,-2.87,-0.29,-2.35,-0.81,0.08,-3.32,1.59,-1.24,0.00,1.25]
print len(scores_total)

#l2_analysis_name_correlBehav = 'l2_analysis_WWW-What_correlBehav-ua_orig-modif_norm'
#l2_analysis_name_correlBehav = 'l2_analysis_WWW-What_correlBehav-swua_orig'

l2_analysis_name_correlBehav = 'l2_analysis_CorrelBehav-scores_total-' + model_name + '-' + preproc_name + '-no_concat_runs'
l2_analysis_name_correlBehav = 'l2_analysis_CorrelBehav-scores_epi-' + model_name + '-' + preproc_name + '-no_concat_runs'
#l2_analysis_name_correlBehav = 'l2_analysis_WWW-What_correlBehav-unwarp_fsl-no_concat_runs'

#group_correl_contrasts = [[('Correl EPI','T', ['mean','scores_epi','scores_total'],[0, 1, 0])],[('Correl Total','T', ['mean','scores_epi','scores_total'],[0, 0, 1])]]
group_correl_contrasts = [[('Correl EPI','T', ['mean','scores_epi'],[0, 1])]]
#group_correl_contrasts = [[('Correl EPI','T', ['mean','scores_total'],[0, 1])]]

numberOfGroupContrastsCorrel = len(group_correl_contrasts)

print "Number of contrasts (Correl): " + str(numberOfGroupContrastsCorrel)

group_correl_contrast_indexes = range(numberOfGroupContrastsCorrel)

#### fonctions utilitaires vite faites ###################################

def get_first(string_list):
    #print string_list
    return string_list[0]
    
def get_second(string_list):
    return string_list[1]
    
### utilitaire pour vérifier la longueur d'une liste sans avoir à créer de nouveaux noeuds
def show_length(files):

    print len(files)
    
    return files
    
### utilitaire pour vérifier une liste sans avoir à créer de nouveaux noeuds
def show_files(files):

    print files
    
    return files

### idem fonction list::index(), mais retourne tous les indices de la liste ou l'item est présent
def get_multiple_indexes(cur_list,item):
    
    return [i for i,list_item in enumerate(cur_list) if list_item == item]

### generate itertools.product in a random order 
import random

def random_product(*args, **kwds):
    "Random selection from itertools.product(*args, **kwds)"
    pools = map(tuple, args) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ############################################################################################################################################################
    ###################################################################### Graph analyses ######################################################################
    ############################################################################################################################################################
    
################################################################## peak activation mask computation #######################################################################

#peak_activation_mask_analysis_name = "compute_peak_activation_mask-all_spm_contrasts"
#peak_activation_mask_analysis_name = "compute_peak_activation_mask_test_from_thr-contrasts_WWW_What"
peak_activation_mask_analysis_name = "compute_peak_activation_mask_test_from_thr"

split_peak_activation_mask_analysis_name = peak_activation_mask_analysis_name.split('_')

#### peak activation ROI definition
### path and contrast images

### from spmT
#spm_contrasts_path = os.path.join(nipype_analyses_path,'l2_analysis-OneSampleTTestModel9_Only3events_Hit-norm_funct-ua_orig-new_st_wbepi-sans_image1-anat-no_concat_runs-anticip',"level2_results_fwe_0_05_topo_fdr_0_05_k_10/l2_contrasts/")
#contrast_pattern = "_contrast_index_[1-6]_group_contrast_index_0/spmT_*.img"

### from spmT_thr
#spm_contrasts_path = os.path.join(nipype_analyses_path,'l2_analysis-OneSampleTTestModel9_Only3events_Hit-norm_funct-ua_orig-new_st_wbepi-sans_image1-anat-no_concat_runs-anticip',"level2_results_fwe_0_05_topo_fdr_0_05_k_10/contrasts_thresh/")
#contrast_pattern = "_contrast_index_[1-6]_group_contrast_index_0/spmT_*_thr.img"

### with spm contrast WWW - What as mask
spm_contrasts_path = os.path.join(nipype_analyses_path,'l2_analysis-OneSampleTTestModel9_Only3events_WWW_What-norm_funct-ua_orig-new_st_wbepi-sans_image1-anat-no_concat_runs-anticip','level2_results_uncor_0_005','contrasts_thresh')
contrast_pattern = "_contrast_index_*_group_contrast_index_0/spmT_*_thr.img"
    
    
### marged mask from multiple contrasts
merged_mask_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name,"merge_spm_mask.nii")

    
# parameters to threshold SPM activation maps
threshold = 7.16
cluster_nbvoxels = 10 # size threshold on bblobs pour Rec_hit

ROI_cube_size = 1 ## neighboorhood expansion (what differences in indexes is a voxel considered a neighbour)
                    ## 1 -> 27 voxels
                    ## 2->125 voxels
#ROI_cube_size = 2

## min number of voxels  de voxels in the neighboorhood
nb_voxels_in_neigh = 27

#min_nb_voxels_in_neigh = 10
min_nb_voxels_in_neigh = 3

min_dist_between_ROIs = (ROI_cube_size*2+1) * np.sqrt(3.0) ### to avoid any shared voxel between ROI cubes 

if model_name == 'WWW_What_model8':
    
    #epi_cond = ['Hit-WWW','Hit-What']# previous version (corresponds to Odor_Hit-WWW Odor_Hit-What in new version)
    epi_cond = ['Odor_Hit-WWW','Odor_Hit-What','Rec_Hit-WWW','Rec_Hit-What','Recall_Hit-WWW','Recall_Hit-What']
    
elif model_name == 'Hit_model8':
    
    epi_cond = ['memory_of_WWW-S3', 'memory_of_What-S3'] 
    #epi_cond = ['memory_of_WWW-S3', 'memory_of_What-S3','memory_of_WWhere-S3','memory_of_WWhich'] 

    
elif model_name == 'Model8_Memory_Hit':
    epi_cond = ['memory_Hit']

elif "Model9" in split_model_name:

    if '3blocs' in split_model_name:
        
        #epi_cond = ['Hit-WWW','Hit-What']# previous version (corresponds to Odor_Hit-WWW Odor_Hit-What in new version)
        epi_cond = ['Odor_Hit-WWW','Odor_Hit-What','Rec_Hit-WWW','Rec_Hit-What','Recall_Hit-WWW','Recall_Hit-What']
        
    elif 'Only3blocs' in split_model_name:
        
        if "WWW" in split_model_name and "What" in split_model_name:
            
            print "Found WWW_What"
            epi_cond = ['Rest_Hit-WWW','Block_Odor_anticipation_Hit-WWW','Block_Odor_preparation_Hit-WWW','Block_Rec_Hit-WWW','Block_Recall_Hit-WWW','Elaboration_Hit-WWW','Rest_Hit-What','Block_Odor_anticipation_Hit-What','Block_Odor_preparation_Hit-What','Block_Rec_Hit-What','Block_Recall_Hit-What','Elaboration_Hit-What']
            
        else:
        
            epi_cond  = ['Rest','Block_Odor_anticipation','Block_Odor_preparation','Block_Rec','Block_Recall','Elaboration']


        
	
    elif 'Only3events' in split_model_name:
        
        
        ### version 5 cond (rest, anticip, odor, rec, recall)
        if "WWW" in split_model_name:
            epi_cond = ['Odor_Hit-WWW','Rec_Hit-WWW','Recall_Hit-WWW','Odor_Hit-What','Rec_Hit-What','Recall_Hit-What']
        
        elif "WWW" in split_model_name and "What" in split_model_name:
            
            print "Found WWW_What"
            epi_cond = ['Rest_Hit-WWW','Odor_anticipation_Hit-WWW','Odor_Hit-WWW','Rec_Hit-WWW','Recall_Hit-WWW','Rest_Hit-What','Odor_anticipation_Hit-What','Odor_Hit-What','Rec_Hit-What','Recall_Hit-What']
            
        elif "Hit" in split_model_name:
            #epi_cond = ['Rest_hit','Odor_anticipation_hit','Odor_hit','Rec_hit','Recall_hit']
            epi_cond = ['Odor_hit','Rec_hit','Recall_hit']
            
        else:
        
            epi_cond = ['Rest','Odor_anticipation','Odor','Rec','Recall']

        #### Older version 6 cond (with prep)
        #if "WWW" in split_model_name and "What" in split_model_name:
            
            #print "Found WWW_What"
            #epi_cond = ['Rest_Hit-WWW','Odor_anticipation_Hit-WWW','Odor_preparation_Hit-WWW','Odor_Hit-WWW','Rec_Hit-WWW','Recall_Hit-WWW','Rest_Hit-What','Odor_anticipation_Hit-What','Odor_preparation_Hit-What','Odor_Hit-What','Rec_Hit-What','Recall_Hit-What']
            
        #else:
        
            #epi_cond = ['Rest','Odor_anticipation','Odor_preparation','Odor','Rec','Recall']


        

    
    
#### new Harvard oxford file, combining cortical and subcortical regions (111 in total) 
full_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "Harvard-Oxford-cortl-sub-recombined-111regions.nii")
resliced_full_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "rHarvard-Oxford-cortl-sub-recombined-111regions.nii")

print resliced_full_HO_img_file

### white matter and ventricule from Harvard oxford 
white_matter_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "Harvard-Oxford-white_matter.nii")
grey_matter_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "Harvard-Oxford-grey_matter.nii")
ventricule_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "Harvard-Oxford-ventricule.nii")

resliced_white_matter_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "rHarvard-Oxford-white_matter.nii")
resliced_grey_matter_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "rHarvard-Oxford-grey_matter.nii")
resliced_ventricule_HO_img_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "rHarvard-Oxford-ventricule.nii")

info_template_file  =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "info-Harvard-Oxford-reorg.txt")

#### save information about peak activation template
### si AAL
#ROI_mask_prefix = "peak_activation_rois-neigh_in_AAL-iter_cluster-nbvoxels_" + str(nb_voxels_in_neigh) + "_dist_peaks_" + str(ROI_cube_size) 

### si HO
ROI_mask_prefix = "peak_activation_rois-neigh_in_HO-iter_cluster-min_nbvoxels_" + str(min_nb_voxels_in_neigh) + "_dist_peaks_" + str(ROI_cube_size) 
#ROI_mask_prefix = "peak_activation_rois-neigh_in_HO-iter_cluster-nbvoxels_" + str(nb_voxels_in_neigh) + "_dist_peaks_" + str(ROI_cube_size) 


#### indexed_mask
indexed_mask_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "indexed_mask-" + ROI_mask_prefix + ".nii")
    
#### saving ROI coords as textfile
### ijk coords
coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-" + ROI_mask_prefix + ".txt")

### coords in MNI space
MNI_coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-MNI-" + ROI_mask_prefix + ".txt")

#### saving ROI coords as textfile
label_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + ".txt")
label_jane_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + "_jane.txt")
    
#### all info in a text file
info_rois_file  =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "info-" + ROI_mask_prefix + ".txt")

#### original spm index for each ROI
orig_spm_index_mask_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "orig_spm_contrast-" + ROI_mask_prefix + ".nii")
    
#### original spm indexes of each ROI (text file)
rois_orig_indexes_file = os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "rois_orig_indexes-" + ROI_mask_prefix + ".txt")
    
    
    ############## when using ROI_coords
    


############## using ROI coords and filtered data

cond = "All"

## cube of side = 3 voxels
#neighbourhood = 1
## cube of side = 5 voxels
neighbourhood = 2

ROI_coords_dir = os.path.join(nipype_analyses_path,"ReseauALS-conj_corPos-OdorRecall")

ROI_coords_labels_file = os.path.join(ROI_coords_dir,"Labels_Network"+cond+".txt")

ROI_coords_MNI_coords_file = os.path.join(ROI_coords_dir,"Coord_Network"+cond+".txt")


ROI_coords_orig_constrast_file = os.path.join(ROI_coords_dir,"Codes_Network"+cond+".txt")


ROI_coords_np_coords_file = os.path.join(ROI_coords_dir,"All_ROI_np_coords.txt")
    
ROI_coords_labelled_mask_file = os.path.join(ROI_coords_dir,"All_labelled_ROI-neigh_"+str(neighbourhood)+".nii")

#### graph analysis with time series extraction from activation peaks (== nodes) #######################

#### weighted correlation analysis
#cor_mat_analysis_name = "Correl_analyses"
#cor_mat_analysis_name = "Correl_analyses-filtered-ROI_peaks-" + model_name + "-cor_mat_analysis_name"
#cor_mat_analysis_name = "Correl_analyses-filtered-ROI_peaks-" + model_name + "-cor_mat_analysis_name-all_spm_contrasts"

cor_mat_analysis_name = "Correl_analyses-filtered-ROI_peaks-" + model_name + "-cor_mat_analysis_name-" + cond

#### min value of BOLD signal to be considered non zeros
min_BOLD_intensity = 50


#### used for computation of the variance
conf_interval_prob = 0.05




######################### timeseries by events ############################

time_series_by_events_analysis = "time_series_by_events_-2_7"





########################## edge dynamics

### number of Kmeans initialisation
nb_Kmeans_init = 1000
    
edge_dyn_dir = os.path.join(nipype_analyses_path,"anticip-events/merge_mask_fwe-T7.16-27vox-143ROIs")

#### graph analysis

### Louvain Traag directory 

louvain_path='/home/david/Packages/Louvain_20110526'
# 
louvain_bin_path=louvain_path + '/bin'

#### Radatools disrectory
radatools_path ='/home/david/Packages/radatools/radatools-3.2-linux32/'

#radatools_comm_path=radatools_path + "/02-Find_Communities"

#### Graph analysis from Z_list/Z_Louvain (now where Z thresholding is defined)

#min_dist_between_voxels = 1.7

#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_signif_conf_correl"

#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_signif_conf_correl-real_optim100"
#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_signif_conf_correl-real_optim100-all_spm_contrasts"
#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_signif_conf_correl-fastoptim-all_spm_contrasts"

#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_correl-fastoptim"
#graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_correl"

graph_analysis_name = "Graph_analysis_" + model_name + "_radatools_correl-" + cond

split_graph_analysis_name = graph_analysis_name.split('_')


### for large networks (heuristics)
#radatools_optim = "WS rfr 1"

### for small networks 
#radatools_optim = "WS trfr 10"

if "fastoptim" in graph_analysis_name.split('-'):
    radatools_optim = "WS rfr 1"
    
else:
    radatools_optim = "WS trfr 100"


######### angles for igraph 3D visualition (now defined in dmgraphanalysis.plot_igraph)

#angle_alpha = 0
#angle_beta = 0

#angle_alpha = 5
#angle_beta = 20

############################################################ Gather / compare partitions (coclass)

########## basic coclass (computation  + all reordering using hclust)

## hierarchical clustering method for reordering coclass matrices
method_hie = 'ward'

#coclass_analysis_name = "Coclass_" + model_name + "_rada_" + cond
#coclass_analysis_name = "Coclass_" + model_name + "_rada_diff_cond_" + cond
coclass_analysis_name = "Coclass_" + model_name + "_rada_diff_event_" + cond

#coclass_analysis_name = "Coclass_" + model_name + "_rada_filter_" + cond
#coclass_analysis_name = "Coclass_" + model_name + "_rada_filter_diff_cond_" + cond

#coclass_analysis_name = "Coclass_" + model_name + "_rada"
#coclass_analysis_name = "Coclass_" + model_name + "_rada_fastoptim"
#coclass_analysis_name = "Coclass_" + model_name + "_rada-all_spm_contrasts"

######### filtering

ROI_coords_filter_file = os.path.join(ROI_coords_dir,"All_ROI_filtered_coclass.txt")




######### forcing



#### compute coclass matrices with force order
#force_order_cond = "Odor_Hit-WWW"

#force_order_file = os.path.join(nipype_analyses_path,"Coclass_" + model_name + "_rada","_cond_" + force_order_cond,"reorder_norm_coclass","node_order_vect.txt")

##force_order_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_order_cond,"reorder_norm_coclass","node_order_vect_ward.txt")

##coclass_analysis_name = "Coclass_forceorder_" + force_order_cond + "_" + model_name + "_rada"

########### sort coclass based on coclassmod

#force_coclassmod = "Odor_Hit-WWW"

##coclass_analysis_name = "Coclass_forcecoclassmod_" + force_coclassmod + "_" + model_name + "_rada"

#lol_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"community_rada_norm_coclass","net_list_norm_coclass_thr_half.lol")
#node_corres_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"prep_rada_norm_coclass","net_list_norm_coclass_thr_half.net")
#coords_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"prep_rada_norm_coclass","net_list_norm_coclass_thr_half.net")

###########

split_coclass_analysis_name = coclass_analysis_name.split('_')






######################################################################## mean correlation

#mean_correl_analysis_name = "MeanCorrel_" + model_name

########### compute mean correlation with forced coclass order 

##mean_correl_analysis_name = "MeanCorrel_forceorder_" + force_order_cond + "_" + model_name + "_rada"

########### sort correlation based on coclassmod

#force_coclassmod = "Odor_Hit-WWW"

##mean_correl_analysis_name = "MeanCorrel_forcecoclassmod_" + force_coclassmod + "_" + model_name + "_rada"

#lol_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"community_rada_norm_coclass","net_list_norm_coclass_thr_half.lol")
#node_corres_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"prep_rada_norm_coclass","net_list_norm_coclass_thr_half.net")
#coords_coclassmod_file = os.path.join(nipype_analyses_path,"Coclass_WWW_What_model8_hie_ward_rada","_cond_" + force_coclassmod,"prep_rada_norm_coclass","net_list_norm_coclass_thr_half.net")

###########

#split_mean_correl_analysis_name = mean_correl_analysis_name.split('_')











########### NBS stats

stat_cor_mat_analysis_name = "Stats_cor_mat_" + model_name 


#stat_analysis_name = "G_coclass_rada_NBS_stats_by_group"
stat_coclass_analysis_name = "Stats_coclass_" + model_name 

#split_stat_analysis_name = stat_analysis_name.split('_')


### value for t-test thresholding

##t_test_thresh= 4.0
#t_test_thresh= 3.7
##t_test_thresh= 3.5
##t_test_thresh= 3.0
t_test_thresh= 2.5 
##t_test_thresh= 2.0

##Set TAIL to left, and thus test the alternative hypothesis that mean of population X < mean of population Y:

#TAIL='both'

### value for binomial CI test 
#conf_interval_binom = 0.0005
conf_interval_binom = 0.001
##conf_interval_binom = 0.005
#conf_interval_binom = 0.01
#conf_interval_binom = 0.05

#Generate 1000 permutations. Many more permutations are required in practice to yield a reliable estimate (e.g. 5000).:

K= 1
#K = 10
#K= 20
#K= 200
#K= 300
#K= 500










###################### for nodewise /pairwise stats

conf_interval_binom_fdr = 0.05
t_test_thresh_fdr = 0.05




















##### the analysis names are further defined depending on the kind of comparisons
############ NBS stats

#### value for t-test thresholding
###THRESH= 4.0
##THRESH= 3.7
###THRESH= 3.5
##THRESH= 3.0
##THRESH= 2.5
##THRESH= 2.0
#THRESH= 1.5

###Set TAIL to left, and thus test the alternative hypothesis that mean of population X < mean of population Y:

#TAIL='both'




#### value for binomial CI test 
###conf_interval_binom = 0.0005
##conf_interval_binom = 0.005
##conf_interval_binom = 0.008
#conf_interval_binom = 0.01
##conf_interval_binom = 0.05

###Generate 1000 permutations. Many more permutations are required in practice to yield a reliable estimate (e.g. 5000).:

##K= 1
###K = 10
##K= 20
##K= 200
###K= 300
#K= 500





