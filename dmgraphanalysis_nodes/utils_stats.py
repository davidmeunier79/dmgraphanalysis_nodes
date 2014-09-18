# -*- coding: utf-8 -*-

import scipy.stats as stat
import numpy as np
import itertools as it

def info_CI(X,Y):
    """ Compute binomial comparaison
"""
    nX = len(X) * 1.
    nY = len(Y) * 1.
    
    pX = np.sum(X == 1)/nX

    pY = np.sum(Y == 1)/nY

    #print pX,pY,np.absolute(pX-pY) 

    SE = np.sqrt(pX * (1-pX)/nX + pY * (1-pY)/nY)

    #print SE
    
    #if (np.absolute(pX-pY) > norm) == True:
        #print pX,pY,np.absolute(pX-pY),norm
   
    
    return np.absolute(pX-pY),SE,np.sign(pX-pY)

    ###################################################################################### pairwise/nodewise stats ###########################################################################################################
    
    
def return_signif_code(p_values,uncor_alpha = 0.05,fdr_alpha = 0.05,bon_alpha = 0.05):

    print p_values
    
    N = p_values.shape[0]
    
    order =  p_values.argsort()
    
    #print order
    
    sorted_p_values= p_values[order]
    
    #print sorted_p_values
    
    ### by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape = N)
    
    ################ uncor #############################
    ### code = 0 for all correlation below uncor_alpha
    signif_code[p_values > uncor_alpha] = 0
    
    ################ FPcor #############################
    
    signif_code[p_values < 1.0/N] = 2
    
    ################ fdr ###############################
    seq = np.arange(N,0,-1)
    
    seq_fdr_p_values = fdr_alpha/seq
    
    #print seq_fdr_p_values
    
    signif_sorted = sorted_p_values < seq_fdr_p_values
    
    signif_code[order[signif_sorted]] = 3
    
    ################# bonferroni #######################
    
    signif_code[p_values < bon_alpha/N] = 4
    
    return signif_code
    
def return_signif_code_Z(Z_values,conf_interval_binom_fdr = 0.05):

    print Z_values
    
    N = Z_values.shape[0]
    
    order =  Z_values.argsort()
    
    
    
    #order =  np_list_diff[:,2].argsort()
    
    #print order
    
    #sort_np_list_diff = np_list_diff[order[::-1]]
    
    
    #print order
    
    sorted_Z_values= Z_values[order[::-1]]
    
    print sorted_Z_values
    
    ### by default, code = 1 (cor at 0.05)
    signif_code = np.ones(shape = N)
    
    ################ uncor #############################
    ### code = 0 for all correlation below uncor_alpha
    Z_uncor = stat.norm.ppf(1-conf_interval_binom_fdr/2)
    
    print Z_uncor
    
    signif_code[Z_values < Z_uncor] = 0
    
    print np.sum(signif_code != 0)
    
    ################ FPcor #############################
    
    Z_FPcor = stat.norm.ppf(1-(1.0/(2*N)))
    
    print Z_FPcor
    
    signif_code[Z_values > Z_FPcor] = 2
    
    print np.sum(signif_code == 2)
    
    ################ fdr ###############################
    
    seq = np.arange(N,0,-1)
    
    seq_fdr_p_values = conf_interval_binom_fdr/seq
    
    seq_Z_val = stat.norm.ppf(1-seq_fdr_p_values/2)
    
    
    #print seq_fdr_p_values
    
    signif_sorted = sorted_Z_values > seq_Z_val
    
    signif_code[order[signif_sorted]] = 3
    
    print np.sum(signif_code == 3)
    
    ################# bonferroni #######################
    
    Z_bon = stat.norm.ppf(1-conf_interval_binom_fdr/(2*N))
    
    print Z_bon
    
    signif_code[Z_values > Z_bon] = 4
    
    print np.sum(signif_code == 4)
    
    return signif_code
    
    
#def return_signif_pval_vect(sort_np_list_diff,t_test_thresh_fdr):

    #n = sort_np_list_diff.shape[0]
    
    ################ uncor #############################
    
    #sort_np_list_diff[sort_np_list_diff[:,1] > t_test_thresh_fdr,2] = 0
    
    ################ fdr ###############################
    #seq = np.arange(n,0,-1)
    
    ##print seq
    
    #seq_fdr_p_values = t_test_thresh_fdr/seq
    
    #sort_np_list_diff[sort_np_list_diff[:,1] < seq_fdr_p_values,2] = sort_np_list_diff[sort_np_list_diff[:,1] < seq_fdr_p_values,2] * 2
    
    ################# bonferroni #######################
    
    #sort_np_list_diff[sort_np_list_diff[:,1] < t_test_thresh_fdr/n,2] = sort_np_list_diff[sort_np_list_diff[:,1] < t_test_thresh_fdr/n,2] * 2
    
    #return sort_np_list_diff
    
    
#def return_signif_pval_mat(np_list_diff,t_test_thresh_fdr):

    #print np_list_diff.shape
    
    #order =  np_list_diff[:,2].argsort()
    
    #print order
    
    #sort_np_list_diff = np_list_diff[order]
    
    #print sort_np_list_diff
    #print sort_np_list_diff.shape[0]
    
    #n = np_list_diff.shape[0]
    
    ################ uncor #############################
    
    #sort_np_list_diff[sort_np_list_diff[:,2] > 0.001,3] = 0
    
    ################ fdr ###############################
    #seq = np.arange(n,0,-1)
    
    ##print seq
    
    #seq_fdr_p_values = t_test_thresh_fdr/seq
    
    #print sort_np_list_diff[:,2] 
    #print seq_fdr_p_values
    
    #sort_np_list_diff[sort_np_list_diff[:,2] < seq_fdr_p_values,3] = sort_np_list_diff[sort_np_list_diff[:,2] < seq_fdr_p_values,3] * 2
    
    ################# bonferroni #######################
    
    #sort_np_list_diff[sort_np_list_diff[:,2] < t_test_thresh_fdr/n,3] = sort_np_list_diff[sort_np_list_diff[:,2] < t_test_thresh_fdr/n,3] * 2
    
    #return sort_np_list_diff
    
    
    
    
    
    
#def return_signif_fdr_values_pval_vect(sort_np_list_diff):

    #n = sort_np_list_diff.shape[0]
    
    #seq = np.arange(n,0,-1)
    
    #print seq
    
    #seq_fdr_p_values = t_test_thresh_fdr/seq
    
    #print np.sum(np.array(sort_np_list_diff[:,1] < seq_fdr_p_values,dtype = 'int'))
    
    #print sort_np_list_diff[sort_np_list_diff[:,1] < seq_fdr_p_values,:]
    
    #return sort_np_list_diff[sort_np_list_diff[:,1] < seq_fdr_p_values,:]
    
#def return_signif_fdr_values_binom_mat(sort_np_list_diff):

    #n = sort_np_list_diff.shape[0]
    
    #seq = np.arange(n,0,-1)
    
    #print seq
    
    #seq_fdr_p_values = conf_interval_binom_fdr/seq
    
    #print seq_fdr_p_values
    ##seq_fdr_p_values
    
    #seq_Z_val = stat.norm.ppf(1-seq_fdr_p_values/2)
    
    #print 
    
    
    #print seq_Z_val
    
    #seq_norm =  Z_val * sort_np_list_diff[:,3]
    
    #print seq_norm
    
    #print np.sum(np.array(sort_np_list_diff[:,2] > seq_norm,dtype = 'int'))
    
    #print sort_np_list_diff[sort_np_list_diff[:,2] > seq_norm,:]
    
    #return sort_np_list_diff[sort_np_list_diff[:,2] > seq_norm,:]
    
    
def compute_pairwise_binom(X,Y,conf_interval_binom):

    # number of nodes
    N = X.shape[0]
   
    # Perform binomial test at each edge
    ADJ = np.zeros((N,N),dtype = 'int')
    
    for i,j in it.combinations(range(N), 2):
        
        ADJ[i,j] = ADJ[j,i] = binom_CI_test(X[i,j,:],Y[i,j,:],conf_interval_binom)
        
    return ADJ

    
def compute_pairwise_ttest_fdr(X,Y,t_test_thresh_fdr):
    
    # number of nodes
    N = X.shape[0]
   
    list_diff = []
    
    for i,j in it.combinations(range(N), 2):
        
        #t_stat_zalewski = ttest2(X[i,j,:],Y[i,j,:])
        
        t_stat,p_val = stat.ttest_ind(X[i,j,:],Y[i,j,:])
        
        #print t_stat,p_val
        
        list_diff.append([i,j,p_val,np.sign(np.mean(X[i,j,:])-np.mean(Y[i,j,:]))])
        
    #print list_diff
        
    np_list_diff = np.array(list_diff)
   
    signif_code = return_signif_code(np_list_diff[:,2],uncor_alpha = 0.001,fdr_alpha = t_test_thresh_fdr, bon_alpha = 0.05)
    
    print np.sum(signif_code == 0.0),np.sum(signif_code == 1.0),np.sum(signif_code == 2.0),np.sum(signif_code == 3.0),np.sum(signif_code == 4.0)
    
    np_list_diff[:,3] = np_list_diff[:,3] * signif_code
    
    print np.sum(np_list_diff[:,3] == 0.0)
    print np.sum(np_list_diff[:,3] == 1.0),np.sum(np_list_diff[:,3] == 2.0),np.sum(np_list_diff[:,3] == 3.0),np.sum(np_list_diff[:,3] == 4.0)
    print np.sum(np_list_diff[:,3] == -1.0),np.sum(np_list_diff[:,3] == -2.0),np.sum(np_list_diff[:,3] == -3.0),np.sum(np_list_diff[:,3] == -4.0)
    
    
    
    
    signif_signed_adj_mat = np.zeros((N,N),dtype = 'int')
    
    signif_i = np.array(np_list_diff[:,0],dtype = int)
    signif_j = np.array(np_list_diff[:,1],dtype = int)
    
    signif_sign = np.array(np_list_diff[:,3],dtype = int)
    
    print signif_i,signif_j
    
    print signif_signed_adj_mat[signif_i,signif_j] 
    
    #print signif_sign
    
    
    
    signif_signed_adj_mat[signif_i,signif_j] = signif_signed_adj_mat[signif_j,signif_i] = signif_sign
    
    print signif_signed_adj_mat
    
    return signif_signed_adj_mat

    
def compute_pairwise_ttest_rel_fdr(X,Y,t_test_thresh_fdr):
    
    # number of nodes
    N = X.shape[0]
   
    list_diff = []
    
    for i,j in it.combinations(range(N), 2):
        
        #t_stat_zalewski = ttest2(X[i,j,:],Y[i,j,:])
        
        t_stat,p_val = stat.ttest_rel(X[i,j,:],Y[i,j,:])
        
        #print t_stat,p_val
        
        list_diff.append([i,j,p_val,np.sign(np.mean(X[i,j,:])-np.mean(Y[i,j,:]))])
        
    #print list_diff
        
    np_list_diff = np.array(list_diff)
   
    signif_code = return_signif_code(np_list_diff[:,2],uncor_alpha = 0.001,fdr_alpha = t_test_thresh_fdr, bon_alpha = 0.05)
    
    print np.sum(signif_code == 0.0),np.sum(signif_code == 1.0),np.sum(signif_code == 2.0),np.sum(signif_code == 3.0),np.sum(signif_code == 4.0)
    
    np_list_diff[:,3] = np_list_diff[:,3] * signif_code
    
    print np.sum(np_list_diff[:,3] == 0.0)
    print np.sum(np_list_diff[:,3] == 1.0),np.sum(np_list_diff[:,3] == 2.0),np.sum(np_list_diff[:,3] == 3.0),np.sum(np_list_diff[:,3] == 4.0)
    print np.sum(np_list_diff[:,3] == -1.0),np.sum(np_list_diff[:,3] == -2.0),np.sum(np_list_diff[:,3] == -3.0),np.sum(np_list_diff[:,3] == -4.0)
    
    
    
    
    signif_signed_adj_mat = np.zeros((N,N),dtype = 'int')
    
    signif_i = np.array(np_list_diff[:,0],dtype = int)
    signif_j = np.array(np_list_diff[:,1],dtype = int)
    
    signif_sign = np.array(np_list_diff[:,3],dtype = int)
    
    #print signif_i,signif_j
    
    #print signif_signed_adj_mat[signif_i,signif_j] 
    
    #print signif_sign
    
    
    
    signif_signed_adj_mat[signif_i,signif_j] = signif_signed_adj_mat[signif_j,signif_i] = signif_sign
    
    #print signif_signed_adj_mat
    
    return signif_signed_adj_mat

    
    
    
def compute_pairwise_binom_fdr(X,Y,conf_interval_binom_fdr):

    # number of nodes
    N = X.shape[0]
   
    # Perform binomial test at each edge
    
    
    list_diff = []
    
    for i,j in it.combinations(range(N), 2):
        
        abs_diff,SE,sign_diff = info_CI(X[i,j,:],Y[i,j,:])
         
        list_diff.append([i,j,abs_diff/SE,sign_diff])
        
    print list_diff
    
    np_list_diff = np.array(list_diff)
    
    signif_code = return_signif_code_Z(np_list_diff[:,2],conf_interval_binom_fdr)
    
    print np.sum(signif_code == 0.0),np.sum(signif_code == 1.0),np.sum(signif_code == 2.0),np.sum(signif_code == 3.0),np.sum(signif_code == 4.0)
    
    np_list_diff[:,3] = np_list_diff[:,3] * signif_code
    
    print np.sum(np_list_diff[:,3] == 0.0)
    print np.sum(np_list_diff[:,3] == 1.0),np.sum(np_list_diff[:,3] == 2.0),np.sum(np_list_diff[:,3] == 3.0),np.sum(np_list_diff[:,3] == 4.0)
    print np.sum(np_list_diff[:,3] == -1.0),np.sum(np_list_diff[:,3] == -2.0),np.sum(np_list_diff[:,3] == -3.0),np.sum(np_list_diff[:,3] == -4.0)
    
    signif_signed_adj_mat = np.zeros((N,N),dtype = 'int')
    
    signif_i = np.array(np_list_diff[:,0],dtype = int)
    signif_j = np.array(np_list_diff[:,1],dtype = int)
    
    signif_sign = np.array(np_list_diff[:,3],dtype = int)
    
    #print signif_i,signif_j
    
    #print signif_signed_adj_mat[signif_i,signif_j] 
    
    #print signif_sign
    
    
    
    signif_signed_adj_mat[signif_i,signif_j] = signif_signed_adj_mat[signif_j,signif_i] = signif_sign
    
    #print signif_signed_adj_mat
    
    return signif_signed_adj_mat

    
    
    
    
#def compute_nodewise_t_values(X,Y):

    ## number of nodes
    #N = X.shape[0]
   
    ## Perform binomial test at each edge
    #t_val_vect = np.zeros((N),dtype = 'float')
    
    #for i in range(N):
        
        #t_val_vect[i] = ttest2(X[i,:],Y[i,:])
        
    #return t_val_vect

    
#def compute_nodewise_t_values_fdr(X,Y,t_test_thresh_fdr):

    ## number of nodes
    #N = X.shape[0]
   
    #list_diff = []
    
    #for i in range(N):
        
        ##t_stat_zalewski = ttest2(X[i,:],Y[i,:])
        
        #t_stat,p_val = stat.ttest_ind(X[i,:],Y[i,:])
        
        #print t_stat,p_val
        
        #list_diff.append([i,p_val,np.sign(np.mean(X[i,:])-np.mean(Y[i,:]))])
        
    ##print list_diff
        
    #np_list_diff = np.array(list_diff)
   
    #print np_list_diff
    
    #sort_np_list_diff = np_list_diff[np_list_diff[:,1].argsort()]
    
    #print sort_np_list_diff.shape[0]
    
    #signif_fdr = return_signif_pval_vect(sort_np_list_diff,t_test_thresh_fdr)
    
    ##print signif_fdr
    
    
    #print np.sum(signif_fdr[:,2] == 0.0),np.sum(signif_fdr[:,2] == 1.0),np.sum(signif_fdr[:,2] == 2.0),np.sum(signif_fdr[:,2] == 4.0),np.sum(signif_fdr[:,2] == -1.0),np.sum(signif_fdr[:,2] == -2.0),np.sum(signif_fdr[:,2] == -4.0)
            
    #signif_signed_vect = np.zeros((N),dtype = 'int')
    
    #signif_signed_vect[np.array(signif_fdr[:,0],dtype = int)] = np.array(signif_fdr[:,2],dtype = int)
    
    #print signif_signed_vect
    
    #return signif_signed_vect

    
    
    
    
#def compute_pairwise_binom_adj_mat(d_stacked, nx, ny):

    #print d_stacked.shape
    
    #assert d_stacked.shape[2] == nx + ny
    
    #t1 = time.time()
    
    #binom_adj_mat = compute_pairwise_binom(d_stacked[:,:,:nx],d_stacked[:,:,nx:nx+ny])

    #t2 = time.time()
    
    #print "computation took %f" %(t2-t1)
    
    #return binom_adj_mat
    
    
#def compute_pairwise_binom_fdr_sign_adj_mat(d_stacked, nx, ny):

    #print d_stacked.shape
    
    #assert d_stacked.shape[2] == nx + ny
    
    #t1 = time.time()
    
    #signif_pos_adj_mat,signif_neg_adj_mat = compute_pairwise_binom_fdr(d_stacked[:,:,:nx],d_stacked[:,:,nx:nx+ny])

    
    #t2 = time.time()
    
    #print "computation took %f" %(t2-t1)
    
    #return  signif_pos_adj_mat,signif_neg_adj_mat
    
    
    
def compute_nodewise_t_test_vect(d_stacked, nx, ny):

    print d_stacked.shape
    
    assert d_stacked.shape[1] == nx + ny
    
    t1 = time.time()
    
    t_val_vect = compute_nodewise_t_values(d_stacked[:,:nx],d_stacked[:,nx:nx+ny])

    t2 = time.time()
    
    print "computation took %f" %(t2-t1)
    
    return t_val_vect
    
######################## correl ######################################

def compute_pairwise_correl_fdr(X,behav_score,correl_thresh_fdr):


    from scipy.stats.stats import pearsonr

    # number of nodes
    N = X.shape[0]
   
    list_diff = []
    
    for i,j in it.combinations(range(N), 2):
        
        #t_stat_zalewski = ttest2(X[i,j,:],Y[i,j,:])
        
        r_stat,p_val = pearsonr(X[i,j,:],behav_score)
        
        #print i,j,p_val,r_stat
        
        list_diff.append([i,j,p_val,np.sign(r_stat)])
        
    #print list_diff
        
    np_list_diff = np.array(list_diff)
   
   
    np_list_diff = np.array(list_diff)
   
    signif_code = return_signif_code(np_list_diff[:,2],uncor_alpha = 0.001,fdr_alpha = correl_thresh_fdr, bon_alpha = 0.05)
    
    print np.sum(signif_code == 0.0),np.sum(signif_code == 1.0),np.sum(signif_code == 2.0),np.sum(signif_code == 3.0),np.sum(signif_code == 4.0)
    
    np_list_diff[:,3] = np_list_diff[:,3] * signif_code
    
    print np.sum(np_list_diff[:,3] == 0.0)
    print np.sum(np_list_diff[:,3] == 1.0),np.sum(np_list_diff[:,3] == 2.0),np.sum(np_list_diff[:,3] == 3.0),np.sum(np_list_diff[:,3] == 4.0)
    print np.sum(np_list_diff[:,3] == -1.0),np.sum(np_list_diff[:,3] == -2.0),np.sum(np_list_diff[:,3] == -3.0),np.sum(np_list_diff[:,3] == -4.0)
    
    
    signif_signed_adj_mat = np.zeros((N,N),dtype = 'int')
    
        
    signif_i = np.array(np_list_diff[:,0],dtype = int)
    signif_j = np.array(np_list_diff[:,1],dtype = int)
    
    signif_sign = np.array(np_list_diff[:,3],dtype = int)
    
    print signif_i,signif_j
    
    print signif_signed_adj_mat[signif_i,signif_j] 
    
    #print signif_sign
    
    
    
    signif_signed_adj_mat[signif_i,signif_j] = signif_signed_adj_mat[signif_j,signif_i] = signif_sign
    
    print signif_signed_adj_mat
    
    return signif_signed_adj_mat