
import matplotlib.pyplot as plt
import pylab as pl
    
import numpy as np

#def plot_cormat(plot_file, cor_mat,list_labels,label_size =2):
    
    #fig1 = plt.figure()
    #ax = fig1.add_subplot(1,1,1)
    
    #im = ax.matshow(cor_mat,interpolation="none")
    ##plt.axis('off')
    
    #[i.set_visible(False) for i in ax.spines.itervalues()]
    
    ##im.set_cmap('binary')
    #im.set_cmap('spectral')
    #### add labels
    
    #if len(list_labels) != 0:
        
        #if len(list_labels) == cor_mat.shape[0]:
            
            #plt.xticks(range(len(list_labels)), list_labels,rotation='vertical', fontsize=label_size)
            #plt.yticks(range(len(list_labels)), list_labels, fontsize=label_size)
            
            #plt.subplots_adjust(top = 0.8)
        #else:
            #print "Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" %(len(list_labels),cor_mat.shape[0])
            
        
    #plt.tick_params(axis='both',          # changes apply to the x-axis
        #which='both',      # both major and minor ticks are affected
        #bottom='off',      # ticks along the bottom edge are off
        #top='off',         # ticks along the top edge are off
        #left = 'off',
        #right = 'off'
        #)
    ##ax.set_xticklabels(['']+labels)
    ##plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)
    
    #fig1.colorbar(im)
    
    #fig1.savefig(plot_file)
    
    #plt.close(fig1)
    ##fig1.close()
    #del fig1
    
def plot_ranged_cormat(plot_file, cor_mat,list_labels, fix_full_range,label_size = 2):
    
    fig1 = plt.figure(frameon=False)
    ax = fig1.add_subplot(1,1,1)
    
    im = ax.matshow(cor_mat,vmin = fix_full_range[0], vmax = fix_full_range[1],interpolation="none")
    #plt.axis('off')
    
    [i.set_visible(False) for i in ax.spines.itervalues()]
    
    #im.set_cmap('binary')
    im.set_cmap('spectral')
    
    ### add labels
    
    if len(list_labels) != 0:
        
        if len(list_labels) == cor_mat.shape[0]:
            
            plt.xticks(range(len(list_labels)), list_labels,rotation='vertical', fontsize=label_size)
            plt.yticks(range(len(list_labels)), list_labels, fontsize=label_size)
            
            plt.subplots_adjust(top = 0.8)
        else:
            print "Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" %(len(list_labels),cor_mat.shape[0])
            
        
    plt.tick_params(axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left = 'off',
        right = 'off'
        )
    #ax.set_xticklabels(['']+labels)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)
    
    fig1.colorbar(im)
    
    fig1.savefig(plot_file)
    
    plt.close(fig1)
    #fig1.close()
    del fig1
    
def plot_int_mat(plot_file, cor_mat,list_labels, fix_full_range,label_size = 2):
    
    fig1 = plt.figure(frameon=False)
    ax = fig1.add_subplot(1,1,1)
    
    cmap= plt.get_cmap('spectral',9) 
    
    im = ax.matshow(cor_mat,vmin = fix_full_range[0], vmax = fix_full_range[1],interpolation="none", cmap= cmap)
    #plt.axis('off')
    
    [i.set_visible(False) for i in ax.spines.itervalues()]
    
    #im.set_cmap('binary')
    #im.set_cmap('spectral')
    
    ### add labels
    
    if len(list_labels) != 0:
        
        if len(list_labels) == cor_mat.shape[0]:
            
            plt.xticks(range(len(list_labels)), list_labels,rotation='vertical', fontsize=label_size)
            plt.yticks(range(len(list_labels)), list_labels, fontsize=label_size)
            
            plt.subplots_adjust(top = 0.8)
        else:
            print "Warning in utils_plot.plot_cormat, incompatible number of labels %d and matrix shape %d" %(len(list_labels),cor_mat.shape[0])
            
        
    plt.tick_params(axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left = 'off',
        right = 'off'
        )
    #ax.set_xticklabels(['']+labels)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=3)
    
    #fig1.colorbar(im, ticks = range(-4,5))
    fig1.colorbar(im, ticks = range(-4,5))
    
    fig1.savefig(plot_file)
    
    plt.close(fig1)
    #fig1.close()
    del fig1
    
#def plot_hist(plot_hist_file,data,nb_bins = 100):
    
    ##fig2 = figure.Figure()
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    #y, x = np.histogram(data, bins = nb_bins)
    #ax.plot(x[:-1],y)
    ##ax.bar(x[:-1],y, width = y[1]-y[0])
    #fig2.savefig(plot_hist_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2
    
def plot_signals(plot_signals_file,signals_matrix,colors = ['blue']):
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    
    #print signals_matrix.shape[0]
    #print len(signals_matrix.shape)
    
    if len(signals_matrix.shape) == 1:
        ax.plot(range(signals_matrix.shape[0]),signals_matrix[:]) 
    
    else:
        for i in range(signals_matrix.shape[0]): 
        
            if len(colors) == signals_matrix.shape[0]:
                print i,colors[i]
                
                ax.plot(range(signals_matrix.shape[1]),signals_matrix[i,:],colors[i]) 
                
            else:
                ax.plot(range(signals_matrix.shape[1]),signals_matrix[i,:])
     
    #ax.plot(,signals_matrix)
    fig2.savefig(plot_signals_file)
    
    plt.close(fig2)
    #fig2.close()
    del fig2
    

def plot_sep_signals(plot_signals_file,signals_matrix,colors = [],range_signal = -1):
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    
    if range_signal == -1:
        
        range_signal = np.amax(signals_matrix) - np.amin(signals_matrix)
    
    #print range_signal
    
    #print signals_matrix.shape
    
    if len(colors) == signals_matrix.shape[0]:
        
        for i in range(signals_matrix.shape[0]): 
            print range_signal*i
            ax.plot(signals_matrix[i,:] + range_signal*i,colors[i]) 
            
    else:

        for i in range(signals_matrix.shape[0]): 
            #print range_signal*i
            ax.plot(signals_matrix[i,:] + range_signal*i) 
            
    #for i in range(signals_matrix.shape[0]): 
        #print range_signal*i
        
        #sep_signals_matrix[i,:] = signals_matrix[i,:] + range_signal*i
        
    #print sep_signals_matrix
    
    #ax.plot(range(signals_matrix.shape[1]),sep_signals_matrix) 
        
    x1,x2,y1,y2 = ax.axis()
    ax.axis((x1,x2,-2.0,y2))

    fig2.savefig(plot_signals_file)
    
    plt.close(fig2)
    #fig2.close()
    del fig2
        
    return
    
#def plot_curve(plot_curve_file,curve_data,x_range,label):
    

    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    
    
    ##print range_signal
    
    #print curve_data.shape
    
    
    #ax.plot(x_range,curve_data, color = "blue",label = label)
            
    #legend = ax.legend(loc='upper center', shadow=True)
    
    #fig2.savefig(plot_curve_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2
    
#def plot_diff_signals(plot_signals_file,signals_matrix):
    

    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    
    
    ##print range_signal
    
    #print signals_matrix.shape
    
    
    #ax.plot(range(-2,signals_matrix.shape[1]-2),signals_matrix[0,:], color = "blue",label = "WWW")
    #ax.plot(range(-2,signals_matrix.shape[1]-2),signals_matrix[1,:], color = "red",label = "What")
            
    #legend = ax.legend(loc='upper center', shadow=True)
    
    #fig2.savefig(plot_signals_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2

#def plot_diff_signals_errorbar(plot_signals_file,signals_matrix,errorbar_matrix):
    

    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    
    
    ##print range_signal
    
    #print signals_matrix.shape
    
    
    #ax.plot(range(-2,signals_matrix.shape[1]-2),signals_matrix[0,:], color = "blue",label = "WWW")
    #ax.plot(range(-2,signals_matrix.shape[1]-2),signals_matrix[1,:], color = "red",label = "What")
            
    #ax.errorbar(range(-2,signals_matrix.shape[1]-2),signals_matrix[0,:],errorbar_matrix[0,:], color = "blue")
    #ax.errorbar(range(-2,signals_matrix.shape[1]-2),signals_matrix[1,:],errorbar_matrix[1,:], color = "red")
    
    
    #legend = ax.legend(loc='upper center', shadow=True)
    
    #fig2.savefig(plot_signals_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2
    
#def plot_sep_diff_signals(plot_signals_file,signals_matrix):
    
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    
    
    #range_signal = np.amax(signals_matrix) - np.amin(signals_matrix)
    
    #print range_signal
    
    #print signals_matrix.shape
    
    
    #for i in range(signals_matrix.shape[0]): 
    
        #if i == 0:
        
            #color = "blue"
        #else :
            #color = "red"
            
            
        #for j in range(signals_matrix.shape[1]): 
    
        ##print range_signal*i
            #ax.plot(signals_matrix[i,j,:] + range_signal*j,color = color) 
        
    ##for i in range(signals_matrix.shape[0]): 
        ##print range_signal*i
        
        ##sep_signals_matrix[i,:] = signals_matrix[i,:] + range_signal*i
        
    ##print sep_signals_matrix
    
    ##ax.plot(range(signals_matrix.shape[1]),sep_signals_matrix) 
        
    
    
    #fig2.savefig(plot_signals_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2
    
    
#def plot_sep_diff_signals_errorbar(plot_signals_file,signals_matrix,errorbar_matrix):
    
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(1,1,1)
    
    
    #range_signal = np.amax(signals_matrix) - np.amin(signals_matrix)
    
    #print range_signal
    
    #print signals_matrix.shape
    
    
    #for i in range(signals_matrix.shape[0]): 
    
        #if i == 0:
        
            #color = "blue"
        #else :
            #color = "red"
            
            
        #for j in range(signals_matrix.shape[1]): 
    
        ##print range_signal*i
            #ax.plot(signals_matrix[i,j,:] + range_signal*j,color = color) 
        
    ##for i in range(signals_matrix.shape[0]): 
        ##print range_signal*i
        
        ##sep_signals_matrix[i,:] = signals_matrix[i,:] + range_signal*i
        
    ##print sep_signals_matrix
    
    ##ax.plot(range(signals_matrix.shape[1]),sep_signals_matrix) 
        
    
    
    #fig2.savefig(plot_signals_file)
    
    #plt.close(fig2)
    ##fig2.close()
    #del fig2
    
#def plot_diff_signals_by_lines(plot_diff_signals_path,signals_matrix):

    #import os
    
    #for j in range(signals_matrix.shape[1]): 
    
        #plot_signals_file = os.path.join(plot_diff_signals_path,"plot_diff_signals_by_ROIs_" + str(j) + ".eps")
        
        #plot_diff_signals(plot_signals_file,signals_matrix[:,j,:])
        

#def plot_diff_signals_errorbar_by_lines(plot_diff_signals_path,signals_matrix,errorbar_matrix):

    #import os
    
    #for j in range(signals_matrix.shape[1]): 
    
        #plot_signals_file = os.path.join(plot_diff_signals_path,"plot_diff_signals_errorbar_by_ROIs_" + str(j) + ".eps")
        
        #plot_diff_signals_errorbar(plot_signals_file,signals_matrix[:,j,:],errorbar_matrix[:,j,:])
        
    
    