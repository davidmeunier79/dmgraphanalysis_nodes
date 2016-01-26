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
print nipype.__version__

import nipype.interfaces.io as nio
import nipype.interfaces.spm as spm
import nipype.interfaces.dcm2nii as dn
import nipype.interfaces.freesurfer as fs    # freesurfer

import nipype.interfaces.matlab as mlab
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

from nipype.interfaces import fsl
fsl.FSLCommand.set_default_output_type('NIFTI')

#import nipype.interfaces.fsl.BET
import nipype.algorithms.rapidart as ra
import nipype.algorithms.modelgen as model
import nipype.pipeline.engine as pe
from nipype.pipeline.engine import Workflow
#from nipype.interfaces.base import Bunch
from nipype.interfaces.utility import IdentityInterface, Function, Select, Rename


import nipype.interfaces.nipy.preprocess as nipy

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
    #main_path = "/home/david/Mount/NEURO011/IRMf/14_fMRI_AVERSION_PNRA_2008/RESULTS/FONCTIONAL_DATA/PREPROCESSING/"
    main_path = "/media/speeddata/Data/Nipype-Aversion/"
    #main_path = "/media/E114-185D/Data/Nipype-Aversion"

matlab_spm_path='/home/david/matlab/spm8'
mlab.MatlabCommand.set_default_paths(matlab_spm_path)

## pour sam

print main_path

#data_path = main_path + "3_MODIFIED_NAME/"
test_data_path = main_path + "TEST_P23/"
#new_data_path = main_path + "REFORMAT_DATA_BY_SUBJ_REMOVE_FIRST_SCANS/"
#new_data_path = main_path + "REFORMAT_DATA_NII_BY_SUBJ/"



## general path
#main_path = "/home/david/Mount/NEURO011/IRMf/14_fMRI_AVERSION_PNRA_2008/RESULTS/FONCTIONAL_DATA/PREPROCESSING/"

dicom_path = os.path.join(main_path,"A_ORIGINAL_DATA")

dcm2nii_config_anat_file = os.path.join(dicom_path,'dcm2nii_config_anat.ini')

dcm2nii_config_funct_file = os.path.join(dicom_path,'dcm2nii_config_funct.ini')

nifti_path = main_path + 'B_NIFTI_DATA'

change_name_nifti_path = os.path.join(main_path,'C_CHANGE_NAME_NIFTI_DATA')

### peut-etre modifié mais pour l'instant tout est fait dans main_path...

nipype_analyses_path = main_path 


############################################################# loop all set of data
cermep_subject_ids =[
'FERLA02151','VATCH01513',
'GALAM02788','KOTRO02812'
,'PARSA02813','CHACA02814'
,'JOUAN02851','RAHSA02853','PLAJA02876','MARSI02877'
,'SAICO02878','REBCO02902','RISLO02903','LEFCO02904','POIAL02941','LELTA02942','MASMA02943','KERFL02983','COUEM02982','RICCE02984','DELAU02996','BONDA02395','MOHME02009','MHIHA03030','DESJU03031','DECCA03032','GIRLI03069','AUBLO03068','GROEM03175','AIMPA01124','CENTR03176']

#behav_subject_ids =['A26','P27','P28','P29','A30','P31','A32','A33','P34','A35','P36','A37']

#behav_subject_ids =[
#'P06','P07',
#'A08','A09'
#,'A10','P11'
#,'P12','P13','A15','A16'
#,'P17','A18','P19','A20','A21','A22','P23','P24','A25','A26','P27','P28','P29','A30','P31',
#'A32','A33','P34','A35','P36','A37'
#]

behav_subject_ids_noA26 =[
'P06','P07',
'A08','A09'
,'A10','P11'
,'P12','P13','A15','A16'
,'P17','A18','P19','A20','A21','A22','P23','P24','A25','P27','P28','P29','A30','P31',
'A32','A33','P34','A35','P36','A37'
]



cond_order = [
['WANTING','LIKING'],['LIKING','WANTING'],
['LIKING','WANTING'],['LIKING','WANTING']
,['LIKING','WANTING'],['WANTING','LIKING']
,['WANTING','LIKING'],['LIKING','WANTING'],['WANTING','LIKING'],['LIKING','WANTING']
,['LIKING','WANTING'],['WANTING','LIKING'],['WANTING','LIKING'],['LIKING','WANTING'],['WANTING','LIKING'],['LIKING','WANTING'],['LIKING','WANTING'],['LIKING','WANTING'],['WANTING','LIKING'],['WANTING','LIKING'],['WANTING','LIKING'],['LIKING','WANTING'],['WANTING','LIKING'],['WANTING','LIKING'],['LIKING','WANTING'],['LIKING','WANTING'],['WANTING','LIKING'],['WANTING','LIKING'],['LIKING','WANTING'],['LIKING','WANTING'],['LIKING','WANTING']]

##### test one set of data only

#cermep_subject_ids =['FERLA02151','RISLO02903']

#cond_order = [['WANTING','LIKING'],['WANTING','LIKING']]

#behav_subject_ids =['P19','P06']

#### session 

session_days = ['1','2']

stim_days = ['O' , 'OI']



##### old directory organisation
#### directory pro/anti aversion 
#pref_dir_aversion = ["PRO","ANTI"]
#### directory day (days are linked to presentation conditions)
#pref_dir_day = ["DAY1","DAY2"]

#### old directory structure
#pref_dir_lik = ["LIKING","WANTING","ANAT"]

### all subjects for directory creation - list of Pro subject and list of anti (aversive) subject list.
##subj_id_aver = [['P23', 'P12', 'P17', 'P19', 'P13', 'P11', 'P06', 'P07'],['A16', 'A21', 'A15', 'A09', 'A18', 'A10', 'A20', 'A22', 'A08']]


#### new directory subjects 
### subject prefix
pref_subj_id = "SUBJECT_"

#### list of all subjects 
#subj_ids = ['P23', 'P12', 'P17', 'P19', 'P13', 'P11', 'P06', 'P07','A16', 'A21', 'A15', 'A09', 'A18', 'A10', 'A20', 'A22', 'A08']

### stim condition day1 = ODOR, day2 = ODOR+IMAGE

#pref_pres_cond = ['O','OI']
pref_pres_cond = ['OI']

### directory liking/wanting
funct_pref_dir_lik = ['LIKING','WANTING']

funct_sessions = [cond + '_' + lik for cond in pref_pres_cond for lik in funct_pref_dir_lik]

#funct_sessions =  ['OI_LIKING']

all_sessions = funct_sessions + ['ANAT']


############################################################## test one set of data only

#cermep_subject_ids = ['FERLA02151']

#behav_subject_ids = ['P06']

#cond_order = [['WANTING','LIKING']]



#cermep_subject_ids = ['VATCH01513']

#behav_subject_ids = ['P07']

#cond_order = [['LIKING','WANTING']]




#cermep_subject_ids = ['FERLA02151','VATCH01513']

#behav_subject_ids = ['P06','P07']

#cond_order = [['WANTING','LIKING'],['LIKING','WANTING']]





#all_sessions =  ['OI_LIKING','ANAT']

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

### test dimensions 
def check_dimensions(indexes,shape):

    if indexes[0] >= 0 and indexes[0] < shape[0] and indexes[1] >= 0 and indexes[1] < shape[1] and indexes[2] >= 0 and indexes[2] < shape[2] :
        return True
    else: 
        return False
        
######################################## Preprocessing parameters ########################################

nb_scans_to_remove = 4

######################################### Name of analysis ##################################################

#### preproc analysis
#analysis_name = 'test_register_skull_strip_struct_to_mean_funct'
#analysis_name = 'test_sam_tout_dans_le_meme_fichier'
#analysis_name = 'test_nipype_0_9'

#analysis_name = 'register_struct_to_mean_funct_nipype_9'
#~ analysis_name = 'register_struct_to_mean_funct_nipype_7'
#analysis_name = 'register_skull_strip_struct_to_mean_funct_nipype_7'


#analysis_name = 'nipy_4D_preprocess_level1'

dartel_analysis_name = 'D_preprocess_dartel'

#preproc_l1_analysis_name = 'D_full_4D_preprocess_level1_funct_to_struct_dartel'
#preproc_analysis_name = 'D_full_4D_preprocess_level1_funct_to_struct'
#preproc_analysis_name = 'D_full_4D_preprocess_struct_to_mean_funct_spm'
preproc_analysis_name = 'D_full_4D_preprocess_struct_to_mean_funct_spm_norm'


TR = 2.5



###################################### reanalyse données JP

preproc_analysis_name = 'D_full_4D_preprocess_struct_to_mean_funct-spm_norm_jp'

split_preproc_analysis_name = preproc_analysis_name.split('-')


loc_swra_path = os.path.join(main_path,'SWRA_Files')

funct_sessions_jp = [lik + '_' + cond for cond in pref_pres_cond for lik in funct_pref_dir_lik]

print funct_sessions_jp

swra_path = '/home/david/Mount/crnldata/crnldata/cmo/homme/IRMf/14_fMRI_AVERSION_PNRA_2008/RESULTS/FONCTIONAL_DATA/PREPROCESSING/4_PREPROCESSED_2014/'

#################### level 1 

extra_bias = 2

#l1_analysis_name = "D_level1_spm_model_deriv1"

#l1_analysis_name = "D_level1_spm_model_extrabias_" + str(extra_bias) + "s" 

#l1_analysis_name = "D_level1_spm_model_duration"
#l1_analysis_name = "D_level1_spm_model_duration_regressmot"

#l1_analysis_name = "D_level1_spm_model_amplitude_noA26_noSign"
l1_analysis_name = "D_level1_spm_model_amplitude_noA26"

#l1_analysis_name = "D_level1_spm_model_amplitude_2ndDeriv_noA26"

split_l1_analysis_name = l1_analysis_name.split('_')

#### second level analysis

### the analysis names are further defined depending on the kind of contrasts




######################################## Contrasts definition #############################################

condition_odors = ['cheese','no_cheese']


#### level_one SPM contrast
#if 'regressmot' in split_l1_analysis_name:
    
    #### regressors vs baseline
    #cont4 = ('cheese vs baseline','T', condition_odors + ['regressmot'],[1,0,0])
    #cont5 = ('no_cheese vs baseline','T', condition_odors + ['regressmot'],[0,1,0])
    
    #### regressors cheese_no_cheese
    ##cont1 = ('active > rest','T', condition_odors+ ['regressmot'],[0.5,0.5,0])
    #cont2 = ('cheese > no_cheese','T', condition_odors+ ['regressmot'],[1,-1,0])
    #cont3 = ('no_cheese > cheese ','T', condition_odors+ ['regressmot'],[-1,1,0])

#else:
    #if 'deriv1' in split_l1_analysis_name:
        #### regressors vs baseline
        #cont4 = ('cheese vs baseline','T', condition_odors,[1,1,0,0])
        #cont5 = ('no_cheese vs baseline','T', condition_odors,[0,0,1,1])

        #### regressors cheese_no_cheese
        ##cont1 = ('active > rest','T', condition_odors,[0.5,0.5])
        #cont2 = ('cheese > no_cheese','T', condition_odors,[1,1,-1,-1])
        #cont3 = ('no_cheese > cheese ','T', condition_odors,[-1,-1,1,1])

    #else:
    
### regressors vs baseline
cont4 = ('cheese vs baseline','T', condition_odors,[1,0])
cont5 = ('no_cheese vs baseline','T', condition_odors,[0,1])

#### regressors cheese_no_cheese
##cont1 = ('active > rest','T', condition_odors,[0.5,0.5])
cont2 = ('cheese > no_cheese','T', condition_odors,[1,-1])
cont3 = ('no_cheese > cheese ','T', condition_odors,[-1,1])

### regressors cheese_no_cheese
#contrasts = [cont2,cont3]

### regressors vs baseline
#contrasts = [cont4,cont5]

### regressors vs baseline + cheese_no_cheese
contrasts = [cont4,cont5,cont2,cont3]

numberOfContrasts = len(contrasts) #number of contrasts you specified in the first level analysis
contrast_indexes = range(1,numberOfContrasts+1)

#### level_two SPM contrast
#cont_group1 = ('Group_1','T', ['Group_{1}'],[1])
#cont_group2 = ('Group_2','T', ['Group_{2}'],[1])
cont_group3 = ('Group_1>Group_2','T',['Group_{1}','Group_{2}'],[1,-1])
cont_group4 = ('Group_2>Group_1','T',['Group_{1}','Group_{2}'],[-1,1])

cont_group1 = ('Group_1','T', ['Group_{1}'],[1])
cont_group2 = ('Group_2','T', ['Group_{2}'],[1])

#group_contrasts = [cont_group3,cont_group4]
group_contrasts = [cont_group3,cont_group4,cont_group1,cont_group2]

numberOfGroupContrasts = len(group_contrasts)
group_contrast_indexes = range(0,numberOfGroupContrasts)


#### level_two SPM contrast correlation

correl_variable = 'score'

#correl_variable = 'RT'

#group_correl_contrasts = [[('Correl EPI','T', ['mean','scores_epi','scores_total'],[0, 1, 0])],[('Correl Total','T', ['mean','scores_epi','scores_total'],[0, 0, 1])]]
group_correl_contrasts = [('Correl ' + correl_variable,'T', ['mean',correl_variable],[0, 1])]

numberOfGroupContrastsCorrel = len(group_correl_contrasts)

print "Number of contrasts (Correl): " + str(numberOfGroupContrastsCorrel)

group_correl_contrast_indexes = range(numberOfGroupContrastsCorrel)











###### special ROI activation extraction (beta values from l1 analysis)
spm_level1_path = os.path.join(nipype_analyses_path,l1_analysis_name,"l1_analysis")

peak_activation_mask_analysis_name = "E_compute_peak_activation_mask_pairwise_cheese-no-cheese_O-OI"

#peak_activation_mask_analysis_name = os.path.join(main_path,"ROIs_Selection3_pour_David")
#peak_activation_mask_analysis_name = os.path.join(main_path,"ROIs_Selection2_pour_David")


neighbourhood = 1 ## neighboorhood expansion (what differences in indexes is a voxel considered a neighbour)
                    ## 1 -> 27 voxels, 2->125 voxels

### min number of voxels  de voxels in the neighboorhood
##nb_voxels_in_neigh = 27

#min_nb_voxels_in_neigh = 10

#min_dist_between_ROIs = (2 * ROI_cube_size + 1) * np.sqrt(3.0) ### to avoid any shared voxel between ROI cubes 

#### min value of BOLD signal to be considered non zeros
min_BOLD_intensity = 50


##### used for computation of the variance
#conf_interval_prob = 0.05

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


##### saving ROI coords as textfile
#### ijk coords
#coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-" + ROI_mask_prefix + ".txt")

#### coords in MNI space
#MNI_coord_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "coords-MNI-" + ROI_mask_prefix + ".txt")

##### saving ROI coords as textfile
#label_rois_file =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "labels-" + ROI_mask_prefix + ".txt")
    
##### all info in a text file
#info_rois_file  =  os.path.join(nipype_analyses_path,peak_activation_mask_analysis_name, "info-" + ROI_mask_prefix + ".txt")


#### labeled_indexed_mask

#### HO_sub
#ROI_dir = os.path.join(nipype_analyses_path,"HO_sub")

### selection2 et selection3

#ROI_dir = os.path.join(nipype_analyses_path,"ROIs_Selection2_pour_David")
#ROI_coords_MNI_coords_file = os.path.join(ROI_dir,"Hypothalamus_Coordinates_MNI.txt")
#ROI_coords_labels_file =  os.path.join(ROI_dir,"Hypothalamus_Names.txt")


#ROI_dir = os.path.join(nipype_analyses_path,"ROIs_Selection1_pour_David")
##ROI_dir = os.path.join(nipype_analyses_path,"ROIs_Selection3_pour_David")

#ROI_mask_file = os.path.join(ROI_dir,"all_ROIs_labelled_mask.nii")
#ROI_label_file = os.path.join(ROI_dir,"labels_all_ROIs.txt")


## clusters

#ROI_dir = os.path.join(nipype_analyses_path,"Coord_ROI")

#ROI_mask_file = os.path.join(ROI_dir,"ROI_coords_mask4.nii")
#ROI_coords_MNI_coords_file = os.path.join(ROI_dir,"CLUSTERS_AMPLI-d1_(-10s)4.txt")
#ROI_coords_labels_file =  os.path.join(ROI_dir,"CLUSTERS_AMPLI-d1_(-10s)_Names4.txt")

### ROI single nifti file

ROI_dir = os.path.join(nipype_analyses_path,"Package6_David")


### correlation

#ROI_dir = os.path.join(nipype_analyses_path,"FunctConnectivity-Coords-retest")

#ROI_mask_file =  os.path.join(ROI_dir,"All_labelled_ROI-neigh_"+str(neighbourhood)+".nii")
#ROI_coords_MNI_coords_file = os.path.join(ROI_dir,"Correlations1_Coordinates.txt")
#ROI_coords_labels_file =  os.path.join(ROI_dir,"Correlations1_Names.txt")

#cor_mat_analysis_name = "Correl_analyses-filtered-ROI_coords-amplitude_noA26-retest"

#ROI_mask_file =  os.path.join(ROI_dir,"All_labelled_ROI2-neigh_"+str(neighbourhood)+".nii")
#ROI_coords_MNI_coords_file = os.path.join(ROI_dir,"Correlations2_Coordinates.txt")
#ROI_coords_labels_file =  os.path.join(ROI_dir,"Correlations2_Names.txt")

#cor_mat_analysis_name = "Correl_analyses-filtered-ROI_coords2-amplitude_noA26-retest"


### 3eme jeu de ROI pour 
ROI_dir = os.path.join(nipype_analyses_path,"FunctConnectivity-Coords3")

ROI_mask_file =  os.path.join(ROI_dir,"All_labelled_ROI-neigh_"+str(neighbourhood)+".nii")

ROI_coords_MNI_coords_file = os.path.join(ROI_dir,"Correlations3_Coordinates.txt")
ROI_coords_labels_file =  os.path.join(ROI_dir,"Correlations3_Names.txt")

cor_mat_analysis_name = "Correl_analyses-filtered-ROI_coords3-amplitude_noA26"


conf_interval_prob = 0.05
    