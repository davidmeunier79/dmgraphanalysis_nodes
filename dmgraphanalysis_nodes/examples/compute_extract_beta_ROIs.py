# -*- coding: utf-8 -*-
"""
extracting beta from level1 analysis (with amplitude)
"""

# generate the dataframe with all result in .h5
import sys, os
sys.path.append('../behaviour')

from pandas.io.pytables import HDFStore

from  define_variables_jp import *
    
from dmgraphanalysis_nodes.labeled_mask import compute_labelled_mask_from_anat_ROIs,compute_ROI_nii_from_ROI_coords_files,compute_recombined_HO_template
from dmgraphanalysis_nodes.labeled_mask import compute_labelled_mask_from_ROI_coords_files
from dmgraphanalysis_nodes.labeled_mask import compute_labelled_mask_from_HO_sub


def compute_beta_2betas(ROI_path,ROI_mask_file,ROI_label_file):

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading ROI mask"
    
    
    if not os.path.exists(ROI_path):
        os.makedirs(ROI_path)
        
    #ROI_path = os.path.join(nipype_analyses_path,"Selection_Anat_ROIs")
    
    #ROI_img_file = os.path.join(ROI_path,"all_anat_ROIs_labelled_mask.nii")
    
    #ROI_label_file = os.path.join(ROI_path,"labels_all_anat_ROIs.txt")
    
    
    print "loading labeled_mask img"
    
    ROI_img = nib.load(ROI_mask_file)
    
    ROI_data = ROI_img.get_data()
    
    print ROI_data.shape
    
    print np.unique(ROI_data)[1:]
    print len(np.unique(ROI_data)[1:])
    
    print "loading ROI labels"
    
    ROI_labels = [line.strip().split()[-1] for line in open(ROI_label_file)]
    
    print len(ROI_labels)
        
    if len(np.unique(ROI_data)[1:]) != len(ROI_labels):
        
        print "$$$$$$$$$$$$$ Warning, error between label names and index in labeled_mask"
        
        sys.exit()
        
    list_reg = ['hrf','d1']
    
    for i in range(len(condition_odors)):
        
        for j,tag in enumerate(list_reg):
        
            beta_ind = i*len(list_reg) + j + 1
            
            print i,j,tag,condition_odors[i], str(beta_ind).zfill(4)
            
            for sess in funct_sessions_jp:
            
                print sess
            
                all_beta_values = []
        
                all_nb_nans = []
                
                all_percent_nans = []
                
                for  subject_num in behav_subject_ids_noA26:
                    
                    #print subject_num
                    
                    beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"level1estimate_amplitude","beta_" + str(beta_ind).zfill(4) + ".img" )
                        
                    #print beta_file
                    
                    beta_img = nib.load(beta_file)
                    
                    beta_data = beta_img.get_data()
                    
                    #print beta_data.shape
                    
                    subj_nb_nans = []
                    
                    subj_percent_nans = []
                    
                    subj_beta_values = []
                    
                    for index in np.unique(ROI_data)[1:]:
                    
                        #print index
                        
                        beta_vals = beta_data[ROI_data == index]
                        
                        #print beta_vals
                        
                        nb_nans = np.sum(np.isnan(beta_vals) == True)
                        
                        #print nb_nans
                        
                        mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                        
                        #print mean_beta_val
                        
                        #if nb_nans != 0:
                        
                            #print beta_vals
                            #print beta_vals[np.logical_not(np.isnan(beta_vals))]
                            #print mean_beta_val
                            
                        subj_beta_values.append(mean_beta_val)
                        
                        subj_nb_nans.append(nb_nans)
                        
                        subj_percent_nans.append(float(nb_nans)/beta_vals.size)
                        
                    all_beta_values.append(subj_beta_values)
                        
                    all_nb_nans.append(subj_nb_nans)
                            
                    all_percent_nans.append(subj_percent_nans)
                            
                np_all_beta_values = np.array(all_beta_values,dtype = 'f')
                
                print np_all_beta_values.shape
                
                df_all_beta_values = pd.DataFrame(np_all_beta_values,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                df_all_beta_values_file = os.path.join(ROI_path,"ROI_coords_" + tag + "_" + sess + '_' + condition_odors[i] +'_beta_values.txt')
                
                df_all_beta_values.to_csv(df_all_beta_values_file)
                
                
                
                
                np_all_nb_nans = np.array(all_nb_nans,dtype = 'int')
                
                print np_all_nb_nans.shape
                
                df_all_nb_nans = pd.DataFrame(np_all_nb_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                df_all_nb_nans_file = os.path.join(ROI_path,"ROI_coords_" + tag + "_" + sess + '_' + condition_odors[i] +'_nb_nans.txt')
                
                df_all_nb_nans.to_csv(df_all_nb_nans_file)
                
                
                np_all_percent_nans = np.array(all_percent_nans,dtype = 'f')
                
                print np_all_percent_nans.shape
                
                df_all_percent_nans = pd.DataFrame(np_all_percent_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                df_all_percent_nans_file = os.path.join(ROI_path,"ROI_coords_"+ tag + "_" + sess + '_' + condition_odors[i] +'_percent_nans.txt')
                
                df_all_percent_nans.to_csv(df_all_percent_nans_file)

def compare_beta_2betas(ROI_path,ROI_mask_file,ROI_label_file):

    import numpy as np
    import pandas as pd
    
    import nibabel as nib
    
    print "Loading ROI mask"
    
    
    #ROI_path = os.path.join(nipype_analyses_path,"Selection_Anat_ROIs")
    
    #ROI_img_file = os.path.join(ROI_path,"all_anat_ROIs_labelled_mask.nii")
    
    #ROI_label_file = os.path.join(ROI_path,"labels_all_anat_ROIs.txt")
    
    ROI_img = nib.load(ROI_mask_file)
    
    ROI_data = ROI_img.get_data()
    
    print ROI_data.shape
    
    print np.unique(ROI_data)[1:]
    print len(np.unique(ROI_data)[1:])
    
    print "loading ROI labels"
    
    ROI_labels = [line.strip().split()[-1] for line in open(ROI_label_file)]
    
    print len(ROI_labels)
        
    for sess in funct_sessions_jp:
    
        print sess
    
        all_beta_values = []

        all_nb_nans = []
        
        all_percent_nans = []
        
        for  subject_num in behav_subject_ids_noA26:
                
                ### hrf and d1 sep 
            list_reg = ['hrf','d1']

            subj_beta_values = []
            
            for i in range(len(condition_odors)):
                
                for j,tag in enumerate(list_reg):
                
                    beta_ind = i*len(list_reg) + j + 1
                    
                    print i,j,tag,condition_odors[i], str(beta_ind).zfill(4)
                    
                    beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"level1estimate_amplitude","beta_" + str(beta_ind).zfill(4) + ".img" )
                        
                    print beta_file
                    
                    beta_img = nib.load(beta_file)
                    
                    beta_data = beta_img.get_data()
                    
                    beta_values = []
                    
                    for index in np.unique(ROI_data)[1:]:
                    
                        beta_vals = beta_data[ROI_data == index]
                        
                        #nb_nans = np.sum(np.isnan(beta_vals) == True)
                        
                        mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                        
                        beta_values.append(mean_beta_val)
                        
                    subj_beta_values.append(beta_values)
                        
            np_subj_beta_values = np.array(subj_beta_values)
            
            print np_subj_beta_values.shape
            
            #### amplitude
            subj_amplitude_values = []
            
            for i in range(len(condition_odors)):
                
                amplitude_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"amplitude","nooffset_beta_" + str(i+1).zfill(4) + ".img" )
                        
                print amplitude_file
                
                amplitude_img = nib.load(amplitude_file)
                
                amplitude_data = amplitude_img.get_data()
                
                amplitude_values = []
                
                for index in np.unique(ROI_data)[1:]:
                
                    amplitude_vals = amplitude_data[ROI_data == index]
                    
                    nb_nans = np.sum(np.isnan(amplitude_vals) == True)
                    
                    mean_amplitude_val = np.mean(amplitude_vals[np.logical_not(np.isnan(amplitude_vals))])
                    
                    amplitude_values.append(mean_amplitude_val)
                    
                subj_amplitude_values.append(amplitude_values)
                    
            np_subj_amplitude_values = np.array(subj_amplitude_values)
            
            print np_subj_amplitude_values.shape
            
            for i in range(len(condition_odors)):
        
                hrf_val = np_subj_beta_values[i*len(list_reg),]
                
                d1_val = np_subj_beta_values[i*len(list_reg)+1,]
                
                comp_ampl_val = np.sign(hrf_val) * np.sqrt(np.square(hrf_val) + np.square(d1_val))
                
                ampl_val = np_subj_amplitude_values[i,]
                
                print ampl_val - comp_ampl_val
                
            0/0
            
            
                        
#def compute_beta_anat_ROIs_amplitude():

    #import numpy as np
    #import pandas as pd
    
    #import nibabel as nib
    
    #print "Loading ROI mask"
    
    
    #ROI_path = os.path.join(nipype_analyses_path,"Selection_Anat_ROIs")
    
    #ROI_img_file = os.path.join(ROI_path,"all_anat_ROIs_labelled_mask.nii")
    
    #ROI_img = nib.load(ROI_img_file)
    
    #ROI_data = ROI_img.get_data()
    
    #print ROI_data.shape
    
    #print "loading ROI labels"
    
    #ROI_label_file = os.path.join(ROI_path,"labels_all_anat_ROIs.txt")
    
    #ROI_labels = [line.strip().split()[-1] for line in open(ROI_label_file)]
    
    #print ROI_labels 
    
    #for i in range(len(condition_odors)):
        
        #for sess in funct_sessions_jp:
        
            #print sess
        
            #all_beta_values = []
    
            #all_nb_nans = []
            
            #all_percent_nans = []
            
            #for  subject_num in behav_subject_ids_noA26:
                
                #print subject_num
                
                #beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"amplitude","beta_" + str(i+1).zfill(4) + ".img" )
                    
                #print beta_file
                
                #beta_img = nib.load(beta_file)
                
                #beta_data = beta_img.get_data()
                
                #print beta_data.shape
                
                #subj_nb_nans = []
                
                #subj_percent_nans = []
                
                #subj_beta_values = []
                
                #for index in np.unique(ROI_data)[1:]:
                
                    ##print index
                    
                    #beta_vals = beta_data[ROI_data == index]
                    
                    ##print beta_vals
                    
                    #nb_nans = np.sum(np.isnan(beta_vals) == True)
                    
                    ##print nb_nans
                    
                    #mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                    
                    ##print mean_beta_val
                    
                    ##if nb_nans != 0:
                    
                        ##print beta_vals
                        ##print beta_vals[np.logical_not(np.isnan(beta_vals))]
                        ##print mean_beta_val
                        
                    #subj_beta_values.append(mean_beta_val)
                    
                    #subj_nb_nans.append(nb_nans)
                    
                    #subj_percent_nans.append(float(nb_nans)/beta_vals.size)
                    
                #all_beta_values.append(subj_beta_values)
                    
                #all_nb_nans.append(subj_nb_nans)
                        
                #all_percent_nans.append(subj_percent_nans)
                        
            #np_all_beta_values = np.array(all_beta_values,dtype = 'f')
            
            #print np_all_beta_values.shape
            
            
            #df_all_beta_values = pd.DataFrame(np_all_beta_values,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_beta_values_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + sess + '_' + condition_odors[i] +'_beta_values.txt')
            
            #df_all_beta_values.to_csv(df_all_beta_values_file)
            
            
            
            
            #np_all_nb_nans = np.array(all_nb_nans,dtype = 'int')
            
            #print np_all_nb_nans.shape
            
            #df_all_nb_nans = pd.DataFrame(np_all_nb_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_nb_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + sess + '_' + condition_odors[i] +'_nb_nans.txt')
            
            #df_all_nb_nans.to_csv(df_all_nb_nans_file)
            
            
            #np_all_percent_nans = np.array(all_percent_nans,dtype = 'f')
            
            #print np_all_percent_nans.shape
            
            #df_all_percent_nans = pd.DataFrame(np_all_percent_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_percent_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + sess + '_' + condition_odors[i] +'_percent_nans.txt')
            
            #df_all_percent_nans.to_csv(df_all_percent_nans_file)
            
            
#def compute_beta_anat_ROIs_2betas():

    #import numpy as np
    #import pandas as pd
    
    #import nibabel as nib
    
    #print "Loading ROI mask"
    
    
    #ROI_path = os.path.join(nipype_analyses_path,"Selection_Anat_ROIs")
    
    #ROI_img_file = os.path.join(ROI_path,"all_anat_ROIs_labelled_mask.nii")
    
    #ROI_img = nib.load(ROI_img_file)
    
    #ROI_data = ROI_img.get_data()
    
    #print ROI_data.shape
    
    #print "loading ROI labels"
    
    #ROI_label_file = os.path.join(ROI_path,"labels_all_anat_ROIs.txt")
    
    #ROI_labels = [line.strip().split()[-1] for line in open(ROI_label_file)]
    
    #print ROI_labels 
    
    #list_reg = ['hrf','d1']
    
    #for i in range(len(condition_odors)):
        
        #for j,tag in enumerate(list_reg):
        
            #beta_ind = i*len(list_reg) + j + 1
            
            #print i,j,tag,condition_odors[i], str(beta_ind).zfill(4)
            
            ##continue
        
                        
            #for sess in funct_sessions_jp:
            
                #print sess
            
                #all_beta_values = []
        
                #all_nb_nans = []
                
                #all_percent_nans = []
                
                #for  subject_num in behav_subject_ids_noA26:
                    
                    #print subject_num
                    
                    #beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"level1estimate_amplitude","beta_" + str(beta_ind).zfill(4) + ".img" )
                        
                    #print beta_file
                    
                    #beta_img = nib.load(beta_file)
                    
                    #beta_data = beta_img.get_data()
                    
                    #print beta_data.shape
                    
                    #subj_nb_nans = []
                    
                    #subj_percent_nans = []
                    
                    #subj_beta_values = []
                    
                    #for index in np.unique(ROI_data)[1:]:
                    
                        ##print index
                        
                        #beta_vals = beta_data[ROI_data == index]
                        
                        ##print beta_vals
                        
                        #nb_nans = np.sum(np.isnan(beta_vals) == True)
                        
                        ##print nb_nans
                        
                        #mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                        
                        ##print mean_beta_val
                        
                        ##if nb_nans != 0:
                        
                            ##print beta_vals
                            ##print beta_vals[np.logical_not(np.isnan(beta_vals))]
                            ##print mean_beta_val
                            
                        #subj_beta_values.append(mean_beta_val)
                        
                        #subj_nb_nans.append(nb_nans)
                        
                        #subj_percent_nans.append(float(nb_nans)/beta_vals.size)
                        
                    #all_beta_values.append(subj_beta_values)
                        
                    #all_nb_nans.append(subj_nb_nans)
                            
                    #all_percent_nans.append(subj_percent_nans)
                            
                #np_all_beta_values = np.array(all_beta_values,dtype = 'f')
                
                #print np_all_beta_values.shape
                
                
                #df_all_beta_values = pd.DataFrame(np_all_beta_values,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_beta_values_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + tag + "_" + sess + '_' + condition_odors[i] +'_beta_values.txt')
                
                #df_all_beta_values.to_csv(df_all_beta_values_file)
                
                
                
                
                #np_all_nb_nans = np.array(all_nb_nans,dtype = 'int')
                
                #print np_all_nb_nans.shape
                
                #df_all_nb_nans = pd.DataFrame(np_all_nb_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_nb_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + tag + "_" + sess + '_' + condition_odors[i] +'_nb_nans.txt')
                
                #df_all_nb_nans.to_csv(df_all_nb_nans_file)
                
                
                #np_all_percent_nans = np.array(all_percent_nans,dtype = 'f')
                
                #print np_all_percent_nans.shape
                
                #df_all_percent_nans = pd.DataFrame(np_all_percent_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_percent_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"Anat_ROIs_" + tag + "_" + sess + '_' + condition_odors[i] +'_percent_nans.txt')
                
                #df_all_percent_nans.to_csv(df_all_percent_nans_file)
           
#def compute_beta_HO_amplitude():

    #import numpy as np
    #import pandas as pd
    
    #import nibabel as nib
    
    #print "Loading HO sub mask"
    
    
    #ROI_path = os.path.join(nipype_analyses_path,"E_compute_peak_activation_mask_pairwise_cheese-no-cheese_O-OI")
    
    #ROI_img_file = os.path.join(ROI_path,"ROI_mask.nii")
    
    #ROI_img = nib.load(ROI_img_file)
    
    #ROI_data = np.array(ROI_img.get_data(),dtype = 'int64')
    
    #print ROI_data.shape
    #print np.unique(ROI_data)
    
    #print "loading HO sub labels"
    
    #label_file = os.path.join(ROI_path,"info-Harvard-Oxford-reorg.txt")
    
    #index_lab = []
    
    #labels = []
    
    #for line in open(label_file):
    
        #splitted_line = line.strip().split()
        
        #index_lab.append(int(splitted_line[0]))
        
        #labels.append(splitted_line[-1])
        
    #print labels
       
    
    #ROI_labels = []
    
    #print np.unique(ROI_data)[1:].tolist()
    
    #for index_ROI in np.unique(ROI_data)[1:]:
        
        #for i in range(len(index_lab)):
        
            #print index_ROI,index_lab[i]
            
            #if index_lab[i] == index_ROI:
            
                #ROI_labels.append(labels[i])
        
    #print ROI_labels 
    
    #for i in range(len(condition_odors)):
        
        #for sess in funct_sessions_jp:
        
            #print sess
        
            #all_beta_values = []
    
            #all_nb_nans = []
            
            #all_percent_nans = []
            
            #for  subject_num in behav_subject_ids_noA26:
                
                #print subject_num
                
                #beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"amplitude","beta_" + str(i+1).zfill(4) + ".img" )
                    
                #print beta_file
                
                #beta_img = nib.load(beta_file)
                
                #beta_data = beta_img.get_data()
                
                #print beta_data.shape
                
                #subj_nb_nans = []
                
                #subj_percent_nans = []
                
                #subj_beta_values = []
                
                #for index in np.unique(ROI_data)[1:]:
                
                    ##print index
                    
                    #beta_vals = beta_data[ROI_data == index]
                    
                    ##print beta_vals
                    
                    #nb_nans = np.sum(np.isnan(beta_vals) == True)
                    
                    ##print nb_nans
                    
                    #mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                    
                    ##print mean_beta_val
                    
                    ##if nb_nans != 0:
                    
                        ##print beta_vals
                        ##print beta_vals[np.logical_not(np.isnan(beta_vals))]
                        ##print mean_beta_val
                        
                    #subj_beta_values.append(mean_beta_val)
                    
                    #subj_nb_nans.append(nb_nans)
                    
                    #subj_percent_nans.append(float(nb_nans)/beta_vals.size)
                    
                #all_beta_values.append(subj_beta_values)
                    
                #all_nb_nans.append(subj_nb_nans)
                        
                #all_percent_nans.append(subj_percent_nans)
                        
            #np_all_beta_values = np.array(all_beta_values,dtype = 'f')
            
            #print np_all_beta_values.shape
            
            
            #df_all_beta_values = pd.DataFrame(np_all_beta_values,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_beta_values_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + sess + '_' + condition_odors[i] +'_beta_values.txt')
            
            #df_all_beta_values.to_csv(df_all_beta_values_file)
            
            
            
            
            #np_all_nb_nans = np.array(all_nb_nans,dtype = 'int')
            
            #print np_all_nb_nans.shape
            
            #df_all_nb_nans = pd.DataFrame(np_all_nb_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_nb_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + sess + '_' + condition_odors[i] +'_nb_nans.txt')
            
            #df_all_nb_nans.to_csv(df_all_nb_nans_file)
            
            
            
            #np_all_percent_nans = np.array(all_percent_nans,dtype = 'float')
            
            #print np_all_percent_nans.shape
            
            #df_all_percent_nans = pd.DataFrame(np_all_percent_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
            
            #df_all_percent_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + sess + '_' + condition_odors[i] +'_percent_nans.txt')
            
            #df_all_percent_nans.to_csv(df_all_percent_nans_file)
            
#def compute_beta_HO_2betas():

    #import numpy as np
    #import pandas as pd
    
    #import nibabel as nib
    
    #print "Loading HO sub mask"
    
    
    #ROI_path = os.path.join(nipype_analyses_path,"E_compute_peak_activation_mask_pairwise_cheese-no-cheese_O-OI")
    
    #ROI_img_file = os.path.join(ROI_path,"ROI_mask.nii")
    
    #ROI_img = nib.load(ROI_img_file)
    
    #ROI_data = np.array(ROI_img.get_data(),dtype = 'int64')
    
    #print ROI_data.shape
    #print np.unique(ROI_data)
    
    #print "loading HO sub labels"
    
    #label_file = os.path.join(ROI_path,"info-Harvard-Oxford-reorg.txt")
    
    #index_lab = []
    
    #labels = []
    
    #for line in open(label_file):
    
        #splitted_line = line.strip().split()
        
        #index_lab.append(int(splitted_line[0]))
        
        #labels.append(splitted_line[-1])
        
    #print labels
       
    #ROI_labels = []
    
    #print np.unique(ROI_data)[1:].tolist()
    
    #for index_ROI in np.unique(ROI_data)[1:]:
        
        #for i in range(len(index_lab)):
        
            #print index_ROI,index_lab[i]
            
            #if index_lab[i] == index_ROI:
            
                #ROI_labels.append(labels[i])
        
    #print ROI_labels 
        
    #list_reg = ['hrf','d1']
    
    #for i in range(len(condition_odors)):
        
        #for j,tag in enumerate(list_reg):
        
            #beta_ind = i*len(list_reg) + j + 1
            
            #print i,j,tag,condition_odors[i], str(beta_ind).zfill(4)
            
            ##continue
        
            #for sess in funct_sessions_jp:
            
                #print sess
            
                #all_beta_values = []
        
                #all_nb_nans = []
                
                #all_percent_nans = []
                
                #for  subject_num in behav_subject_ids_noA26:
                    
                    ##print subject_num
                    
                    #beta_file = os.path.join(nipype_analyses_path,l1_analysis_name,"l1analysis","_session_" + sess + "_subject_id_" + subject_num,"level1estimate_amplitude","beta_" + str(beta_ind).zfill(4) + ".img" )
                        
                    ##print beta_file
                    
                    #beta_img = nib.load(beta_file)
                    
                    #beta_data = beta_img.get_data()
                    
                    ##print beta_data.shape
                    
                    #subj_nb_nans = []
                    
                    #subj_percent_nans = []
                    
                    #subj_beta_values = []
                    
                    #for index in np.unique(ROI_data)[1:]:
                    
                        ##print index
                        
                        #beta_vals = beta_data[ROI_data == index]
                        
                        ##print beta_vals
                        
                        #nb_nans = np.sum(np.isnan(beta_vals) == True)
                        
                        ##print nb_nans
                        
                        #mean_beta_val = np.mean(beta_vals[np.logical_not(np.isnan(beta_vals))])
                        
                        ##print mean_beta_val
                        
                        ##if nb_nans != 0:
                        
                            ##print beta_vals
                            ##print beta_vals[np.logical_not(np.isnan(beta_vals))]
                            ##print mean_beta_val
                            
                        #subj_beta_values.append(mean_beta_val)
                        
                        #subj_nb_nans.append(nb_nans)
                        
                        #subj_percent_nans.append(float(nb_nans)/beta_vals.size)
                        
                    #all_beta_values.append(subj_beta_values)
                        
                    #all_nb_nans.append(subj_nb_nans)
                            
                    #all_percent_nans.append(subj_percent_nans)
                            
                #np_all_beta_values = np.array(all_beta_values,dtype = 'f')
                
                #print np_all_beta_values.shape
                
                
                #df_all_beta_values = pd.DataFrame(np_all_beta_values,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_beta_values_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + tag + "_" + sess + '_' + condition_odors[i] +'_beta_values.txt')
                
                #df_all_beta_values.to_csv(df_all_beta_values_file)
                
                
                
                
                #np_all_nb_nans = np.array(all_nb_nans,dtype = 'int')
                
                #print np_all_nb_nans.shape
                
                #df_all_nb_nans = pd.DataFrame(np_all_nb_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_nb_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + tag + "_" + sess + '_' + condition_odors[i] +'_nb_nans.txt')
                
                #df_all_nb_nans.to_csv(df_all_nb_nans_file)
                
                #np_all_percent_nans = np.array(all_percent_nans,dtype = 'float')
                
                #print np_all_percent_nans.shape
                
                #df_all_percent_nans = pd.DataFrame(np_all_percent_nans,columns = ROI_labels,index = behav_subject_ids_noA26)
                
                #df_all_percent_nans_file = os.path.join(nipype_analyses_path,l1_analysis_name,"HO_sub_" + tag + "_" + sess + '_' + condition_odors[i] +'_percent_nans.txt')
                
                #df_all_percent_nans.to_csv(df_all_percent_nans_file)
           
if __name__ =='__main__':
    
    
    #ref_img = "/media/speeddata/Data/Nipype-Aversion/D_l2_analysis_contrast_pro_anti_amplitude_noA26_cheeseNocheese/level2_results_thr_0_001/contrasts_thresh/ravg152T2.nii"
    
    #resliced_full_HO_img_file,info_template_file = compute_recombined_HO_template(ref_img,ROI_dir)
    
    ### from HO sub template 
    #ROI_mask_file,ROI_label_file = compute_labelled_mask_from_HO_sub(resliced_full_HO_img_file,info_template_file,ROI_dir)
    #compute_beta_2betas(ROI_dir,ROI_mask_file,ROI_label_file)
    
    ### from a list of VOI single files
    #compute_ROI_nii_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file,ROI_coords_labels_file,neighbourhood = neighbourhood)
    
    ROI_mask_file,ROI_label_file = compute_labelled_mask_from_anat_ROIs(resliced_full_HO_img_file,ROI_dir)
    
    ##compute_beta_2betas(ROI_dir,ROI_mask_file,ROI_label_file)
    #compute_beta_2betas_by_ROI(ROI_dir,ROI_mask_file,ROI_label_file)
    
    
    ## from a list
    
    #ROI_mask_file = compute_labelled_mask_from_ROI_coords_files(resliced_full_HO_img_file,ROI_coords_MNI_coords_file)
    
    #compute_beta_2betas(ROI_dir,ROI_mask_file,ROI_coords_labels_file)
    
    
    
    
    ######### old
    ###### extract beta values from ROIs
    #compute_beta_anat_ROIs_2betas()
    #compute_beta_HO_2betas()
    
    #compute_beta_anat_ROIs_amplitude()
    #compute_beta_HO_amplitude()

    ### test, comparing amplitude values from sum hrf + d1 and amplidude function
    #compare_beta_2betas(ROI_dir,ROI_mask_file,ROI_label_file)