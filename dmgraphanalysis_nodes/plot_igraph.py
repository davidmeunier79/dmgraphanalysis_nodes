# -*- coding: utf-8 -*-
"""
Support function for igraph
"""

import numpy as np
import os


import igraph as ig

import cairo as ca

import math

from dmgraphanalysis_nodes.utils_igraph import igraph_colors,project2D_np

from dmgraphanalysis_nodes.utils_dtype_coord import where_in_coords,find_index_in_coords
from dmgraphanalysis_nodes.utils_igraph import add_non_null_labels,return_base_weighted_graph


def plot_3D_igraph_int_mat(plot_nbs_adj_mat_file,int_matrix,coords = np.array([]),labels = [], edge_colors = ['Gray','Blue','Red']):
    
    g = return_base_weighted_graph(int_matrix)
    
    print labels
    
    if len(labels) == len(g.vs):
    
        add_non_null_labels(g,labels)
        
    
    vertex_degree = np.array(g.degree())*0.2
    
    print np.unique(int_matrix)
    
    for i,index in enumerate(np.unique(int_matrix)[1:]):
    
        colored_egde_list = g.es.select(weight_eq = index)
        
        print len(colored_egde_list),np.sum(int_matrix == index)
        
        for e in colored_egde_list:
        
            print e.tuple
            
            print int_matrix[e.tuple[0],e.tuple[1]]        
        
        colored_egde_list["color"] = edge_colors[i]
    
        print i,index,len(colored_egde_list)
        
        
    
    
    
    if coords.shape[0] != len(g.vs):
    
        layout2D = g.layout_fruchterman_reingold()
    
    else:
    
        layout2D = project2D_np(coords).tolist()
        
    
    ###print g
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
    #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
    ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D , vertex_size = vertex_degree,    edge_curved = False)
    
    return plot_nbs_adj_mat_file
    
    #def plot_igraph_3D_int_mat(int_matrix,coords,plot_nbs_adj_mat_file):
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #mod_list = int_matrix.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(mod_list,mode=ig.ADJ_MAX)
        
        #vertex_degree = np.array(g.degree())*0.2
        
        
        
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        
        
    #def plot_igraph_3D_int_mat(int_matrix,coords,plot_nbs_adj_mat_file):
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #mod_list = int_matrix.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(mod_list,mode=ig.ADJ_MAX)
        
        #vertex_degree = np.array(g.degree())*0.2
        
        
        
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        
        
        
        
        
    #def plot_igraph_3D_int_upper_mat(upper_matrix,coords,plot_nbs_adj_mat_file, labels = []):
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #g= ig.Graph.Weighted_Adjacency(upper_matrix.tolist(),mode=ig.ADJ_UPPER, loops = False)
        
        #vertex_degree = np.array(g.degree())*0.2
        
        
        
        
        #if len(labels) == len(g.vs):
        
            #g.vs['label'] = labels
            
            #g.vs['label_size'] = 5
        
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = False)
        
        
        
        
    #def plot_igraph_3D_signed_bin_label_mat(int_matrix,coords,plot_nbs_adj_mat_file,labels = []):
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #mod_list = int_matrix.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(mod_list,mode=ig.ADJ_MAX)
        
        #print len(labels),len(g.vs)
        
        #print g.degree()
        
        #null_degree_index, = np.where(np.array(g.degree()) == 0)
        
        #print null_degree_index
        
        #np_labels = np.array(labels,dtype = 'string')
        
        #np_labels[null_degree_index] = ""
        
        #print np_labels
        
        #if len(labels) == len(g.vs):
        
            #g.vs['label'] = np_labels.tolist()
            
            #g.vs['label_size'] = 15
        
        #print len(g.es)
        
        #if len(g.es) > 0 :
            
            ##print g.es['weight']
            
            #edge_col = []
            
            #for w in g.es['weight']:
                
                ##(e0,e1) = e.tuple
                
                ##print int(e.weight)
                
                ##comp_index = int(e.weight)
                
                #if int(w) == -1:
                    #edge_col.append('green')
                #elif int(w) == -2:
                    #edge_col.append('cyan')
                #elif int(w) == -3:
                    #edge_col.append('blue')
                #elif int(w) == -4:
                    #edge_col.append('darkblue')
                    
                #elif int(w) == 1:
                    #edge_col.append('yellow')
                #elif int(w) == 2:
                    #edge_col.append('orange')
                #elif int(w) == 3:
                    #edge_col.append('darkorange')
                #elif int(w) == 4:
                    #edge_col.append('red')
                    
            
            ##g_all.es['names'] = edge_list_names
            ##g_all.vs['names'] = node_list_names
            
            #g.es['color'] = edge_col
            
            #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = 0.2,    edge_width =  1)
            
        #else:
            #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = 0.2,    edge_width =  0.01)
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        
        
        
    #def plot_igraph_3D_categ_upper_mat(upper_matrix,coords,plot_nbs_adj_mat_file,labels = [],dict_colors = igraph_colors):
        
        #import cairo
        
        #print upper_matrix.shape
        #print coords.shape 
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #g= ig.Graph.Weighted_Adjacency(upper_matrix.tolist(),mode=ig.ADJ_UPPER, loops = False)
        
        ##print g.es['weight']
        
        #edge_col = []
        
        ##dict_colors[13] = 'grey'
        
        
        #for w in g.es['weight']:
            
            ##(e0,e1) = e.tuple
            
            ##print int(e.weight)
            
            ##comp_index = int(e.weight)
            
            #if int(w) < nb_igraph_colors:
                #edge_col.append(dict_colors[int(w)])
                
                ##print dict_colors[int(w)],int(w)
                
            #else:
                #edge_col.append("lightgrey")
                
                
        ##print edge_col
        
        #g.es['color'] = edge_col
        
        #### labels 
        #null_degree_index, = np.where(np.array(g.degree()) == 0)
        
        #print null_degree_index
        
        #np_labels = np.array(labels,dtype = 'string')
        
        #np_labels[null_degree_index] = ""
        
        #print np_labels
        
        #if len(labels) == len(g.vs):
        
            #g.vs['label'] = np_labels.tolist()
            
            #g.vs['label_size'] = 10
        
        
        #print g.vs
        
        ##vertex_degree = np.array(g.degree())*0.2
        
            
            
        #####print g
        ###ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        ###ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        #plot = ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = 1.0,    edge_width = 1)
        
        #### Construct the plot
        ##plot = ig.Plot(plot_nbs_adj_mat_file, bbox=(600, 650), background="white")

        ### Grab the surface, construct a drawing context and a TextDrawer
        ##ctx = cairo.Context(plot.surface)
        ##ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        
        ### Create the graph and add it to the plot
        ##plot.add(g, bbox=(20, 70, 580, 630), layout = layout2D.tolist() , vertex_size = 1.0,    edge_width = 1)

        ### Make the plot draw itself on the Cairo surface
        ##plot.redraw()

        ### Save the plot
        ##plot.save()


        
    #def plot_igraph_3D_int_label_mat(int_matrix,coords,plot_nbs_adj_mat_file):
        
        #layout2D = project2D_np(coords)
        
        ##print layout2D
            
        #mod_list = int_matrix.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(mod_list,mode=ig.ADJ_MAX)
        
        
        ##print g.es['weight']
        
        #edge_col = []
        
        #for w in g.es['weight']:
            
            ##(e0,e1) = e.tuple
            
            ##print int(e.weight)
            
            ##comp_index = int(e.weight)
            
            #if int(w) < nb_igraph_colors:
                #edge_col.append(igraph_colors[int(w)])
                
            #else:
                #edge_col.append("lightgrey")
                
        
        ##g_all.es['names'] = edge_list_names
        ##g_all.vs['names'] = node_list_names
        
        #g.es['color'] = edge_col
        
        
        
        #vertex_degree = np.array(g.degree())*0.2
        
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = np.array(g.es['weight']), edge_curved = True)
        ##ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01)
        
        
        
        
        
        
        
        
        
        
    #def plot_igraph_3D_adj_mat(adj_matrix,coords,plot_nbs_adj_mat_file):
        
        #layout2D = project2D_np(coords_list)
        
        ##print layout2D
            
        #mod_list = adj_matrix.tolist()
        
        #g= ig.Graph.Adjacency(mod_list,mode=ig.ADJ_MAX)
        
        #vertex_degree = np.array(g.degree())*0.2
        
        ##print vertex_degree
        
        ##g.es['sign'] = np.sign(g)
        
        ##
        ####print g
        #ig.plot(g, plot_nbs_adj_mat_file, layout = layout2D.tolist() , vertex_size = vertex_degree,    edge_width = 0.01, edge_curved = True)
        
    #def plot_igraph_3D_avg_cor_mat(avg_cor_mat,coords):
        
        ##layout2D = project2D(node_coords.tolist(),0,0)
        #layout2D = project2D_np(coords)
        
        ##print mod_cor_mat
        
        #wei_sig_list = avg_cor_mat.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(wei_sig_list,mode=ig.ADJ_MAX)
        
        ##print g
        
        ##g.es['sign'] = np.sign(g)
        
        
        
        #pos_list = g.es.select(weight_gt = 0)
        
        ##print pos_list
        
        #neg_list = g.es.select(weight_lt = 0)
        
        ##pos_list["color"] = 'red'
        #neg_list["color"] = 'blue'
        
        
        ##pos_list["edge_width"] = 0.01
        #neg_list["edge_width"] = 0.01
        
        #g.delete_edges(pos_list)
        
        #i_graph_file = os.path.abspath('plot_igraph-neg_correl.eps')
        
        
        ##g.vs['color'] = igraph_colors[:node_coords.shape[0]]
        
        ##vertex_degree = np.array(g.degree())*0.2
        
        
        ####print g
        ##ig.plot(g,i_graph_file,layout = layout2D.tolist(),vertex_size = vertex_degree,  edge_width = 0.01, edge_curved = True)
        #ig.plot(g,i_graph_file,layout = layout2D.tolist(),vertex_size = 0.1)
        
        
        
        
        #return i_graph_file
        
        
        
        
    #def plot_3D_igraph_weighted_signed_matrix(wei_sig_mat,node_coords):
        
        ##layout2D = project2D(node_coords.tolist(),0,0)
        #layout2D = project2D_np(node_coords)
        
        ##print mod_cor_mat
        
        #wei_sig_list = wei_sig_mat.tolist()
        
        ##print mod_list
        
        #g= ig.Graph.Weighted_Adjacency(wei_sig_list,mode=ig.ADJ_MAX)
        
        ##print g
        
        ##g.es['sign'] = np.sign(g)
        
        
        
        #pos_list = g.es.select(weight_gt = 0)
        
        ##print pos_list
        
        #neg_list = g.es.select(weight_lt = 0)
        
        #pos_list["color"] = 'red'
        #neg_list["color"] = 'blue'
        
        
        
        #g.vs['color'] = igraph_colors[:node_coords.shape[0]]
        
        #i_graph_file = os.path.abspath('plot_weighted_signed_graph.eps')
        
        ####print g
        #ig.plot(g,i_graph_file,layout = layout2D.tolist())
        
        #return i_graph_file
        
    #def plot_3D_igraph_modules_net_list(community_vect,node_coords,net_list,gm_mask_coords,labels = []):

        #if (community_vect.shape[0] != node_coords.shape[0]):
            #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            
        #find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
        
        #find_in_gm = find_index_in_coords(node_coords,gm_mask_coords)
        
        
        #print np.min(find_in_gm),np.max(find_in_gm),find_in_gm.shape
        
        #np_labels = np.array(labels, dtype = 'string')
        
        #cur_labels = np_labels[find_in_gm]
        
        #print cur_labels
        
        #print np.min(find_in_corres),np.max(find_in_corres),find_in_corres.shape
        
        ############ extract edge list (with coords belonging to )
        
        #edge_list = []
        
        #edge_weights = []
        
        #for u,v,w in net_list:
            ##print u,v
            #if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
            ##and u > v:
                
                #edge_list.append((find_in_corres[u-1],find_in_corres[v-1]))
                #edge_weights.append(w)
                
        #node_list_names = map(str,range(len(community_vect)))
        
        #layout2D = project2D_np(node_coords)
        
        ####################################################### All modules 
        
        #net_list_all_modules_file = os.path.abspath("net_list_all_modules.eps")

        #g= ig.Graph(edge_list)
        
        #g.es["weight"] = edge_weights
        
        ##print g
        
        ########## colors
        
        #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
        #edge_col = []
        
        #for e in g.es:
            
            #(e0,e1) = e.tuple
            
            #if (community_vect[e0] == community_vect[e1]):
                #edge_col.append(igraph_colors[community_vect[e0]])
                
            #else:
                #edge_col.append("lightgrey")
                
        
        #vertex_col = []
        
        #for i,v in enumerate(g.vs):
            #mod_index = community_vect[i]
            #if (mod_index != len(igraph_colors)-1):
                #vertex_col.append(igraph_colors[mod_index])
            #else:
                #vertex_col.append("lightgrey")
        
        #g.vs['color'] = vertex_col
        
        #g.es['color'] = edge_col
        
        #g['layout'] = layout2D.tolist()
        
        
        
        #if len(cur_labels) == len(g.vs):
        
            #print "$$$$$$$$$$$$$$$$ Labels $$$$$$$$$$$$$$$$$$$$$$$$"
            
            #g.vs['label'] = cur_labels
            
            #g.vs['label_size'] = 5
        
        
        
        #ig.plot(g, net_list_all_modules_file, vertex_size = 5, edge_width = 0.01, edge_curved = False)
        ##ig.plot(g_all, net_list_all_modules_file, vertex_size = np.array(g_all.degree())*0.2,    edge_width = np.array(edge_weights)*0.1, edge_curved = False)
        
        #return net_list_all_modules_file
        
        
    #def plot_3D_igraph_modules_node_roles(community_vect,node_coords,Z_list,gm_mask_coords,node_roles):
        
        
        #if (community_vect.shape[0] != node_coords.shape[0]):
            #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            
        #if (community_vect.shape[0] != node_roles.shape[0]):
            #print "Warning, community_vect {} != node_roles {}".format(community_vect.shape[0], node_roles.shape[0])
            

        
        #find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
        
        ############ threshoding the number of dictictly displayed modules with the number of igraph colors
        
        #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
        ############ extract edge list (with coords belonging to )
        
        #edge_col_inter = []
        #edge_list_inter = []
        #edge_weights_inter = []
        
        #edge_col_intra = []
        #edge_list_intra = []
        #edge_weights_intra = []
        
        #for u,v,w in Z_list:
            ##print u,v
            #if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
            ##and u > v:
                
                #e0 = find_in_corres[u-1]
                #e1 = find_in_corres[v-1]
                
                #if (community_vect[e0] == community_vect[e1]):
                    
                    #edge_list_intra.append((e0,e1))
                    #edge_weights_intra.append(w)
                    #edge_col_intra.append(igraph_colors[community_vect[e0]])
                #else:
                    
                    #edge_list_inter.append((e0,e1))
                    #edge_weights_inter.append(w)
                    #edge_col_inter.append("lightgrey")
                    
                
        #layout2D = project2D_np(node_coords)
        
        ####################################################### All modules 
        
        #node_roles_modules_file = os.path.abspath("node_roles_modules.eps")

        #edge_list = edge_list_inter + edge_list_intra
        
        #edge_weights = edge_weights_inter + edge_weights_intra 
        
        #edge_col = edge_col_inter + edge_col_intra
        
        
        #g_all= ig.Graph(edge_list)
        
        #g_all.es["weight"] = edge_weights
        
        #g_all.es['color'] = edge_col
        ##print g_all
        
        ########## colors (module belonging) and shape - size (roles)
        
        #vertex_col = []
        
        #vertex_shape = []
        
        #vertex_size = []
        
        #print len(g_all.vs),node_roles.shape[0]
        
        #for i,v in enumerate(g_all.vs):
            
            #### color
            #mod_index = community_vect[i]
            #if (mod_index != len(igraph_colors)-1):
                #vertex_col.append(igraph_colors[mod_index])
            #else:
                #vertex_col.append("lightgrey")
                
            #### node size
            #if node_roles[i,0] == 1:
                #vertex_size.append(5.0)
            #elif node_roles[i,0] == 2:
                #vertex_size.append(10.0)
                
                
            #### shape
            #if node_roles[i,1] == 1:
                #vertex_shape.append("circle")
                
            #elif node_roles[i,1] == 2 or node_roles[i,1] == 5:
                
                #vertex_shape.append("rectangle")
        
            #elif node_roles[i,1] ==  3 or node_roles[i,1] == 6:
                
                #vertex_shape.append("triangle-up")
                
            #elif node_roles[i,1] == 4 or node_roles[i,1] == 7:
                #vertex_shape.append("triangle-down")
                
        
        #g_all.vs['color'] = vertex_col
        
        #g_all['layout'] = layout2D.tolist()
        
        #ig.plot(g_all, node_roles_modules_file, vertex_size = np.array(vertex_size),    edge_width = np.array(edge_weights)*0.001, edge_curved = False, vertex_shape = vertex_shape)
        
        #return node_roles_modules_file
        
    #def plot_3D_igraph_modules_coomatrix(community_vect,coomatrix,node_coords,node_labels):


        #if community_vect.shape[0] != node_coords.shape[0]:
            #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            
        #if community_vect.shape[0] != node_labels.shape[0]:
            #print "Warning, community_vect {} != node_labels {}".format(community_vect.shape[0], node_labels.shape[0])
        
        #if coomatrix.shape[0] != community_vect.shape[0] or coomatrix.shape[1] != community_vect.shape[0]:
            #print "Warning, community_vect {} != coomatrix {}".format(community_vect.shape[0], coomatrix.shape[0])
            #print "OR, coomatrix is not a square matrix {} != ".format(coomatrix.shape[0],coomatrix.shape[1])
        
    
        #print community_vect.shape,coomatrix.shape,node_coords.shape,node_labels.shape
        
        ############ extract edge list (with coords belonging to )
        
        #edge_col_inter = []
        #edge_list_inter = []
        #edge_weights_inter = []
        
        #edge_col_intra = []
        #edge_list_intra = []
        #edge_weights_intra = []
        
        #for e0,e1,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
        
            #if (community_vect[e0] == community_vect[e1]):
                
                #edge_list_intra.append((e0,e1))
                #edge_weights_intra.append(w)
                #edge_col_intra.append(igraph_colors[community_vect[e0]])
                
                
            #else:
                
                #edge_list_inter.append((e0,e1))
                #edge_weights_inter.append(w)
                #edge_col_inter.append("lightgrey")
                
            
        ####################################################### All modules 
        
        #edge_list = edge_list_intra
        
        #edge_weights = edge_weights_intra 
        
        #edge_col = edge_col_intra
        
        #edge_list = edge_list_inter + edge_list_intra
        
        #edge_weights = edge_weights_inter + edge_weights_intra 
        
        #edge_col = edge_col_inter + edge_col_intra
        
        
        
        #g_all= ig.Graph(edge_list)
        
        #print len(g_all.vs),len(g_all.es)
        
        #g_all.es["weight"] = edge_weights
        
        #g_all.es['color'] = edge_col
        ##print g_all
        
        ########## colors
        
        #vertex_col = []
        
        #for i,v in enumerate(g_all.vs):
            #mod_index = community_vect[i]
            #if (mod_index != len(igraph_colors)-1):
                #vertex_col.append(igraph_colors[mod_index])
            #else:
                #vertex_col.append("lightgrey")
        
        #g_all.vs['color'] = vertex_col
        
        #print len(node_labels),len(g_all.vs)
        
        ##print len(cur_labels),len(g_all.vs)
        
        #if node_labels.shape[0] == len(g_all.vs):
        
            #print "$$$$$$$$$$$$$$$$ Labels $$$$$$$$$$$$$$$$$$$$$$$$"
            
            #g_all.vs['label'] = node_labels.tolist()
            
            #g_all.vs['label_size'] = 5
            
            #g_all.vs['label_dist'] = 2
        
        #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
        
        #suf = ["_from_left","_from_front","_from_top","_from_behind"]
        
        #Z_list_all_modules_files = []
        
        #for i,view in enumerate(views):
        
            #print view
            
            #Z_list_all_modules_file = os.path.abspath("All_modules" + suf[i] + ".eps")

            #Z_list_all_modules_files.append(Z_list_all_modules_file)
            
            #layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
            
            #g_all['layout'] = layout2D.tolist()
        
            ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = np.array(edge_weights)*0.001, edge_curved = False)
            
            #ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
            
        #return Z_list_all_modules_files
        
        
        
        
        
        
        
        
        
        
        #### previous version
    ##def plot_3D_igraph_modules_coomatrix(community_vect,node_coords,coomatrix,gm_mask_coords,labels = []):

        ##if (community_vect.shape[0] != node_coords.shape[0]):
            ##print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            

        ##find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
        
        ##print np.min(find_in_corres),np.max(find_in_corres),find_in_corres.shape
        
        ##find_in_gm = find_index_in_coords(node_coords,gm_mask_coords)
        
        ###print find_in_gm
        ##print np.min(find_in_gm),np.max(find_in_gm),find_in_gm.shape
            
        ##if len(labels) != 0:
                
            ##np_labels = np.array(labels, dtype = 'string')
            
            ##print np_labels
            
            ##cur_labels = np_labels[find_in_gm]
            
            ##print cur_labels
            
        ##else :
            ##cur_labels = np.empty((0), dtype = 'string')
                
        ##print len(labels),cur_labels.shape
        
        ##0/0
        
        ############# extract edge list (with coords belonging to )
        
        ##edge_col_inter = []
        ##edge_list_inter = []
        ##edge_weights_inter = []
        
        ##edge_col_intra = []
        ##edge_list_intra = []
        ##edge_weights_intra = []
        
        ##for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
            ###print u,v
            ##if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
            ###and u > v:
                
                ##e0 = find_in_corres[u-1]
                ##e1 = find_in_corres[v-1]
                
                ##if (community_vect[e0] == community_vect[e1]):
                    
                    ##edge_list_intra.append((e0,e1))
                    ##edge_weights_intra.append(w)
                    ##edge_col_intra.append(igraph_colors[community_vect[e0]])
                    
                    
                ##else:
                    
                    ##edge_list_inter.append((e0,e1))
                    ##edge_weights_inter.append(w)
                    ##edge_col_inter.append("lightgrey")
                    
                
        ######################################################## All modules 
        
        ##edge_list = edge_list_intra
        
        ##edge_weights = edge_weights_intra 
        
        ##edge_col = edge_col_intra
        
        ##edge_list = edge_list_inter + edge_list_intra
        
        ##edge_weights = edge_weights_inter + edge_weights_intra 
        
        ##edge_col = edge_col_inter + edge_col_intra
        
        
        
        ##g_all= ig.Graph(edge_list)
        
        ##g_all.es["weight"] = edge_weights
        
        ##g_all.es['color'] = edge_col
        ###print g_all
        
        ########### colors
        
        ##vertex_col = []
        
        ##for i,v in enumerate(g_all.vs):
            ##mod_index = community_vect[i]
            ##if (mod_index != len(igraph_colors)-1):
                ##vertex_col.append(igraph_colors[mod_index])
            ##else:
                ##vertex_col.append("lightgrey")
        
        ##g_all.vs['color'] = vertex_col
        
        
        
        ##print len(cur_labels),len(g_all.vs)
        
        ##0/0
        ###print len(cur_labels),len(g_all.vs)
        
        ##if len(cur_labels) == len(g_all.vs):
        
            ##print "$$$$$$$$$$$$$$$$ Labels $$$$$$$$$$$$$$$$$$$$$$$$"
            
            ##g_all.vs['label'] = cur_labels
            
            ##g_all.vs['label_size'] = 5
            
            ##g_all.vs['label_dist'] = 2
        
        
        
        ##views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
        
        ##suf = ["_from_left","_from_front","_from_top","_from_behind"]
        
        ##Z_list_all_modules_files = []
        
        ##for i,view in enumerate(views):
        
            ##print view
            
            ##Z_list_all_modules_file = os.path.abspath("All_modules" + suf[i] + ".eps")

            ##Z_list_all_modules_files.append(Z_list_all_modules_file)
            
            ##layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
            
            ##g_all['layout'] = layout2D.tolist()
        
            ###ig.plot(g_all, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = np.array(edge_weights)*0.001, edge_curved = False)
            
            ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
            
        ##return Z_list_all_modules_files
        
        
    #def plot_3D_igraph_single_modules_coomatrix(community_vect,node_coords,coomatrix,gm_mask_coords):

        #if (community_vect.shape[0] != node_coords.shape[0]):
            #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            
        #find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
        
        ############ threshoding the number of dictictly displayed modules with the number of igraph colors
        
        #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
        ############ extract edge list (with coords belonging to )
        
        #print np.unique(community_vect)
        
        #Z_list_all_modules_files = []
            
        #for mod_index in np.unique(community_vect):
        
            
            #edge_col_intra = []
            #edge_list_intra = []
            #edge_weights_intra = []
            
            #for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
                ##print u,v
                #if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
                ##and u > v:
                    
                    #e0 = find_in_corres[u-1]
                    #e1 = find_in_corres[v-1]
                    
                    #if (community_vect[e0] == community_vect[e1] and community_vect[e0] == mod_index):
                        
                        #edge_list_intra.append((e0,e1))
                        #edge_weights_intra.append(w)
                        #edge_col_intra.append(igraph_colors[community_vect[e0]])
                        
                        
                    ##else:
                        
                        ##edge_list_inter.append((e0,e1))
                        ##edge_weights_inter.append(w)
                        ##edge_col_inter.append("lightgrey")
                        
                    
            ####################################################### All modules 
            
            #edge_list = edge_list_intra
            
            #edge_weights = edge_weights_intra 
            
            #edge_col = edge_col_intra
            
            ##edge_list = edge_list_inter + edge_list_intra
            
            ##edge_weights = edge_weights_inter + edge_weights_intra 
            
            ##edge_col = edge_col_inter + edge_col_intra
            
            
            
            #g_all= ig.Graph(edge_list)
            
            #g_all.es["weight"] = edge_weights
            
            #g_all.es['color'] = edge_col
            ##print g_all
            
            ########## colors
            
            #vertex_col = []
            
            #for i,v in enumerate(g_all.vs):
                #mod_index = community_vect[i]
                #if (mod_index != len(igraph_colors)-1):
                    #vertex_col.append(igraph_colors[mod_index])
                #else:
                    #vertex_col.append("lightgrey")
            
            #g_all.vs['color'] = vertex_col
            
            
            
            #view = [0.0,0.0]
            
            ##views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
            
            ##suf = ["_from_left","_from_front","_from_top","_from_behind"]
            
            ##for i,view in enumerate(views):
            
            #print view
            
            #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + "_from_left.eps")

            #Z_list_all_modules_files.append(Z_list_all_modules_file)
            
            #layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
            
            #g_all['layout'] = layout2D.tolist()
        
            #ig.plot(g_all, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = np.array(edge_weights)*0.001, edge_curved = False)
            
        #return Z_list_all_modules_files
        
        
    #def plot_3D_igraph_modules_Z_list(community_vect,node_coords,Z_list,gm_mask_coords):

        #if (community_vect.shape[0] != node_coords.shape[0]):
            #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
            
        #find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
        
        ############ threshoding the number of dictictly displayed modules with the number of igraph colors
        
        #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
        ############ extract edge list (with coords belonging to )
        
        #edge_col_inter = []
        #edge_list_inter = []
        #edge_weights_inter = []
        
        #edge_col_intra = []
        #edge_list_intra = []
        #edge_weights_intra = []
        
        #for u,v,w in Z_list:
            ##print u,v
            #if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
            ##and u > v:
                
                #e0 = find_in_corres[u-1]
                #e1 = find_in_corres[v-1]
                
                #if (community_vect[e0] == community_vect[e1]):
                    
                    #edge_list_intra.append((e0,e1))
                    #edge_weights_intra.append(w)
                    #edge_col_intra.append(igraph_colors[community_vect[e0]])
                #else:
                    
                    #edge_list_inter.append((e0,e1))
                    #edge_weights_inter.append(w)
                    #edge_col_inter.append("lightgrey")
                    
                
        #layout2D = project2D_np(node_coords)
        
        ####################################################### All modules 
        
        #Z_list_all_modules_file = os.path.abspath("Z_list_all_modules.eps")

        #edge_list = edge_list_inter + edge_list_intra
        
        #edge_weights = edge_weights_inter + edge_weights_intra 
        
        #edge_col = edge_col_inter + edge_col_intra
        
        
        #g_all= ig.Graph(edge_list)
        
        #g_all.es["weight"] = edge_weights
        
        #g_all.es['color'] = edge_col
        ##print g_all
        
        ########## colors
        
        #vertex_col = []
        
        #for i,v in enumerate(g_all.vs):
            #mod_index = community_vect[i]
            #if (mod_index != len(igraph_colors)-1):
                #vertex_col.append(igraph_colors[mod_index])
            #else:
                #vertex_col.append("lightgrey")
        
        #g_all.vs['color'] = vertex_col
        
        #g_all['layout'] = layout2D.tolist()
        
        #ig.plot(g_all, Z_list_all_modules_file, vertex_size = np.array(g_all.degree())*0.5,    edge_width = np.array(edge_weights)*0.001, edge_curved = False)
        
        #return Z_list_all_modules_file
        
        
    
#def plot_3D_igraph_modules_Z_list(community_vect,node_coords,Z_list,gm_mask_coords):

    #if (community_vect.shape[0] != node_coords.shape[0]):
        #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
        
    #find_in_corres = find_index_in_coords(gm_mask_coords,node_coords)
    
    ############ extract edge list (with coords belonging to )
    
    #edge_list = []
    
    #edge_weights = []
    
    #for u,v,w in Z_list:
        ##print u,v
        #if find_in_corres[u-1] != -1 and find_in_corres[v-1] != -1:
        ##and u > v:
            
            #edge_list.append((find_in_corres[u-1],find_in_corres[v-1]))
            #edge_weights.append(w)
            
    #layout2D = project2D_np(node_coords)
    
    ####################################################### All modules 
    
    #Z_list_all_modules_file = os.path.abspath("Z_list_all_modules.eps")

    #g_all= ig.Graph(edge_list)
    
    #g_all.es["weight"] = edge_weights
    
    ##print g_all
    
    ########## colors
    
    #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
    
    #edge_col = []
    
    #edge_col
    
    #for e in g_all.es:
        
        #(e0,e1) = e.tuple
        
        #if (community_vect[e0] == community_vect[e1]):
            #edge_col.append(igraph_colors[community_vect[e0]])
            
        #else:
            #edge_col.append("lightgrey")
            
    
    #vertex_col = []
    
    #for i,v in enumerate(g_all.vs):
        #mod_index = community_vect[i]
        #if (mod_index != len(igraph_colors)-1):
            #vertex_col.append(igraph_colors[mod_index])
        #else:
            #vertex_col.append("lightgrey")
    
    #g_all.vs['color'] = vertex_col
    
    #g_all.es['color'] = edge_col
    
    #g_all['layout'] = layout2D.tolist()
    
    #ig.plot(g_all, Z_list_all_modules_file, vertex_size = np.array(g_all.degree())*0.5,    edge_width = np.array(edge_weights)*0.001, edge_curved = False)
    
    #return Z_list_all_modules_file
        
    
################################# using relative coords directly 

from dmgraphanalysis_nodes.utils_igraph import add_non_null_labels,add_vertex_colors,create_module_edge_list

def plot_3D_igraph_all_modules(community_vect,Z_list,node_coords = np.array([]),node_labels = [], layout = ''):

    if (community_vect.shape[0] != Z_list.shape[0] or community_vect.shape[0] != Z_list.shape[1]):
        print "Warning, community_vect {} != Z_list {}".format(community_vect.shape[0], Z_list.shape)
    
    ######### creating from coomatrix and community_vect
    g_all = create_module_edge_list(Z_list,community_vect,list_colors = igraph_colors)
    
    ######### vertex colors
    
    add_vertex_colors(g_all,community_vect,list_colors = igraph_colors)
    
    if len(node_labels) != 0:
    
        print "non void labels found"
        
        add_non_null_labels(g_all,node_labels)
        
    else :
        print "empty labels"
        
    if layout == 'FR':
    
        print "plotting with Fruchterman-Reingold layout"
    
        layout2D = g.layout_fruchterman_reingold()
    
        g_all['layout'] = layout2D.tolist()
    
        
        Z_list_all_modules_file = os.path.abspath("All_modules_FR.eps")

        #ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        ig.plot(g_all, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
        
        return [Z_list_all_modules_file]
    
    
    
    
    else:
            
        if node_coords.size != 0:
            
            print "non void coords found"
            
            views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
            
            suf = ["_from_left","_from_front","_from_top","_from_behind"]
            
            Z_list_all_modules_files = []
            
            for i,view in enumerate(views):
            
                print view
                
                Z_list_all_modules_file = os.path.abspath("All_modules_3D" + suf[i] + ".eps")

                Z_list_all_modules_files.append(Z_list_all_modules_file)
                
                layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
                
                g_all['layout'] = layout2D.tolist()
            
                #ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
                ig.plot(g_all, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
                
            return Z_list_all_modules_files
                    
        else :
            
            print "Warning, should have coordinates, or specify layout = 'FR' (Fruchterman-Reingold layout) in options"

def plot_3D_igraph_single_modules(community_vect,coomatrix,node_coords = np.array([]),node_labels = [],nb_min_nodes_by_module = 100):
    
    
    import collections
    
    dist_com = collections.Counter(community_vect)
    
    print dist_com
    
    if (community_vect.shape[0] != node_coords.shape[0]):
        print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
        
    print community_vect.shape
    print node_coords.shape
    print coomatrix.shape
    
    ########### threshoding the number of dictictly displayed modules with the number of igraph colors
    
    community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
    print np.unique(community_vect)
    
    Z_list_all_modules_files = []
    
    ########### extract edge list (with coords belonging to )
    
    g_all = ig.Graph(zip(coomatrix.row, coomatrix.col), directed=False, edge_attrs={'weight': coomatrix.data})
    
    print g_all
    
    for mod_index in np.unique(community_vect):
    
        print "Module index %d has %d nodes"%(mod_index,np.sum(community_vect == mod_index))
        
        if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:
            
            print "Not enough nodes (%d), skipping plot"%(np.sum(community_vect == mod_index))
            continue
        
        g_sel = g_all.copy()
        
        edge_mod_id = []
        edge_col_intra = []
        
        print g_sel
        
        for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
            
            if (community_vect[u] == community_vect[v] and community_vect[u] == mod_index):
                
                edge_col_intra.append(igraph_colors[community_vect[u]])
            else: 
            
                eid = g_sel.get_eid(u,v)
                
                edge_mod_id.append(eid)
                
        g_sel.delete_edges(edge_mod_id)
        
        g_sel.es['color'] = edge_col_intra
        
        ######### node colors
        
        vertex_col = []
        
        for i,v in enumerate(g_sel.vs):
            cur_mod_index = community_vect[i]
            
            if (mod_index == cur_mod_index):
                vertex_col.append(igraph_colors[mod_index])
            else:
                vertex_col.append("black")
        
        g_sel.vs['color'] = vertex_col
        
        
        #### node_labels 
        add_non_null_labels(g_sel,node_labels)
    
        ## single view
        view = [0.0,0.0]
        
        Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + "_from_left.eps")

        layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
        
        g_sel['layout'] = layout2D.tolist()
    
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = 0.1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
        
        Z_list_all_modules_files.append(Z_list_all_modules_file)
        
        #### all view
        #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
        
        #suf = ["_from_left","_from_front","_from_top","_from_behind"]
        
        #Z_list_all_modules_files = []
        
        #for i,view in enumerate(views):
        
            #print view
            
            #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + suf[i] + ".eps")

            #layout2D = project2D_np(node_coords, angle_alpha = view[0],angle_beta = view[1])
            
            #g_sel['layout'] = layout2D.tolist()
        
            ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
            #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
            
            #Z_list_all_modules_files.append(Z_list_all_modules_file)
            
        
    return Z_list_all_modules_files
    
    
            
#def plot_3D_igraph_all_modules_coomatrix_rel_coords(community_vect,node_rel_coords,coomatrix, node_labels = []):
    
    #if (community_vect.shape[0] != node_rel_coords.shape[0]):
        #print "Warning, community_vect {} != node_rel_coords {}".format(community_vect.shape[0], node_rel_coords.shape[0])
        
    #print community_vect.shape
    #print node_rel_coords.shape
    
    #print coomatrix.shape
    
    ############ threshoding the number of dictictly displayed modules with the number of igraph colors
    
    #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
    
    ############ extract edge list (with coords belonging to )
    
    #print np.unique(community_vect)
    
    #edge_col_inter = []
    #edge_list_inter = []
    #edge_weights_inter = []
    
    #edge_col_intra = []
    #edge_list_intra = []
    #edge_weights_intra = []
    
    #for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
        
        #if (community_vect[u] == community_vect[v]):
            
            #edge_list_intra.append((u,v))
            #edge_weights_intra.append(w)
            #edge_col_intra.append(igraph_colors[community_vect[u]])
        #else:
            
            #edge_list_inter.append((u,v))
            #edge_weights_inter.append(w)
            #edge_col_inter.append("lightgrey")
            
            
    #edge_list = edge_list_inter + edge_list_intra
    
    #edge_weights = edge_weights_inter + edge_weights_intra 
    
    #edge_col = edge_col_inter + edge_col_intra
    
    #g_all= ig.Graph(edge_list)
    
    #g_all.es["weight"] = edge_weights
    
    #g_all.es['color'] = edge_col
    ##print g_all
    
    ########## colors
    
    #add_vertex_colors(g_all,community_vect,list_colors = igraph_colors)
    
    #add_non_null_labels(g_all,node_labels)
    
    
    
    #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
    
    #suf = ["_from_left","_from_front","_from_top","_from_behind"]
    
    #Z_list_all_modules_files = []
    
    #for i,view in enumerate(views):
    
        #print view
        
        #Z_list_all_modules_file = os.path.abspath("All_modules" + suf[i] + ".eps")

        #Z_list_all_modules_files.append(Z_list_all_modules_file)
        
        #layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])
        
        #g_all['layout'] = layout2D.tolist()
    
        ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        #ig.plot(g_all, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
        
    #return Z_list_all_modules_files
    
def plot_3D_igraph_single_modules_coomatrix_rel_coords(community_vect,node_rel_coords ,coomatrix, node_labels = [],nb_min_nodes_by_module = 100):
    
    
    import collections
    
    dist_com = collections.Counter(community_vect)
    
    print dist_com
    
    if (community_vect.shape[0] != node_rel_coords.shape[0]):
        print "Warning, community_vect {} != node_rel_coords {}".format(community_vect.shape[0], node_rel_coords.shape[0])
        
    print community_vect.shape
    print node_rel_coords.shape
    print coomatrix.shape
    
    ########### threshoding the number of dictictly displayed modules with the number of igraph colors
    
    community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
        
    print np.unique(community_vect)
    
    Z_list_all_modules_files = []
    
    ########### extract edge list (with coords belonging to )
    
    g_all = ig.Graph(zip(coomatrix.row, coomatrix.col), directed=False, edge_attrs={'weight': coomatrix.data})
    
    #### node_labels 
    add_non_null_labels(g_all,node_labels)
    
    print g_all
    
    for mod_index in np.unique(community_vect):
    
        print "Module index %d has %d nodes"%(mod_index,np.sum(community_vect == mod_index))
        
        if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:
            
            print "Not enough nodes (%d), skipping plot"%(np.sum(community_vect == mod_index))
            continue
        
        g_sel = g_all.copy()
        
        edge_mod_id = []
        edge_col_intra = []
        
        print g_sel
        
        for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
            
            if (community_vect[u] == community_vect[v] and community_vect[u] == mod_index):
                
                edge_col_intra.append(igraph_colors[community_vect[u]])
            else: 
            
                eid = g_sel.get_eid(u,v)
                
                edge_mod_id.append(eid)
                
        g_sel.delete_edges(edge_mod_id)
        
        g_sel.es['color'] = edge_col_intra
        
        ######### node colors
        
        vertex_col = []
        
        for i,v in enumerate(g_sel.vs):
            cur_mod_index = community_vect[i]
            
            if (mod_index == cur_mod_index):
                vertex_col.append(igraph_colors[mod_index])
            else:
                vertex_col.append("black")
        
        g_sel.vs['color'] = vertex_col
        
        ## single view
        view = [0.0,0.0]
        
        Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + "_from_left.eps")

        layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])
        
        g_sel['layout'] = layout2D.tolist()
    
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = 0.1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
        ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
        
        Z_list_all_modules_files.append(Z_list_all_modules_file)
        
        #### all view
        #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
        
        #suf = ["_from_left","_from_front","_from_top","_from_behind"]
        
        #Z_list_all_modules_files = []
        
        #for i,view in enumerate(views):
        
            #print view
            
            #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + suf[i] + ".eps")

            #layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])
            
            #g_sel['layout'] = layout2D.tolist()
        
            ##ig.plot(g_all, Z_list_all_modules_file, vertex_size = 5,    edge_width = 1, edge_curved = False)
            #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 1,    edge_width = 0.1, edge_curved = False)
            
            #Z_list_all_modules_files.append(Z_list_all_modules_file)
            
        
    return Z_list_all_modules_files
    
    
############# with node roles
#def plot_3D_igraph_single_modules_coomatrix_rel_coords_node_roles(community_vect,node_rel_coords,coomatrix,node_roles,nb_min_nodes_by_module = 100):
    
    #import collections
    
    #dist_com = collections.Counter(community_vect)
    
    #print dist_com
    
    #if (community_vect.shape[0] != node_rel_coords.shape[0]):
        #print "Warning, community_vect {} != node_rel_coords {}".format(community_vect.shape[0], node_rel_coords.shape[0])
        
    #print community_vect.shape
    #print node_rel_coords.shape
    
    #print coomatrix.shape
    
    ############ threshoding the number of dictictly displayed modules with the number of igraph colors
    
    #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
    
    
    #dist_com = collections.Counter(community_vect)
    
    #print dist_com
    
    ############ extract edge list (with coords belonging to )
    
    #print np.unique(community_vect)
    
    #Z_list_all_modules_files = []
    
    #mat = coomatrix.todense()
    
    #mat = mat + np.transpose(mat)
    
    #print np.min(mat), np.max(mat)
    
    #g_all= ig.Graph.Weighted_Adjacency(matrix = mat.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight",loops = False)
        
    #for mod_index in np.unique(community_vect):
    
        #print "Module index %d has %d nodes"%(mod_index,np.sum(community_vect == mod_index))
        
        #if np.sum(community_vect == mod_index) < nb_min_nodes_by_module:
            
            #print "Not enough nodes (%d), skipping plot"%(np.sum(community_vect == mod_index))
            #continue
        
        #g_sel = g_all.copy()
        
        #edge_mod_id = []
        #edge_col_intra = []
        
        #for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
            
            #if (community_vect[u] == community_vect[v] and community_vect[u] == mod_index):
                
                #edge_col_intra.append(igraph_colors[community_vect[u]])
            #else: 
                #eid = g_sel.get_eid(u,v)
                
                #edge_mod_id.append(eid)
                
        #g_sel.delete_edges(edge_mod_id)
        
        #g_sel.es['color'] = edge_col_intra
        
        ########## node colors and roles
        
        #vertex_col = []
            
        #vertex_shape = []
        
        #vertex_size = []
        
                
        #for i,v in enumerate(g_sel.vs):
            #cur_mod_index = community_vect[i]
            
            #if (mod_index == cur_mod_index):
                #vertex_col.append(igraph_colors[mod_index])
                    
                    
                #### node size
                #if node_roles[i,0] == 1:
                    #vertex_size.append(5.0)
                #elif node_roles[i,0] == 2:
                    #vertex_size.append(10.0)
                    
                #### shape
                #if node_roles[i,1] == 1:
                    #vertex_shape.append("circle")
                    
                #elif node_roles[i,1] == 2 or node_roles[i,1] == 5:
                    
                    #vertex_shape.append("rectangle")
            
                #elif node_roles[i,1] ==  3 or node_roles[i,1] == 6:
                    
                    #vertex_shape.append("triangle-up")
                    
                #elif node_roles[i,1] == 4 or node_roles[i,1] == 7:
                    #vertex_shape.append("triangle-down")
                    
        
            #else:
                #vertex_col.append("black")
                #vertex_size.append(2.0)
                #vertex_shape.append("circle")
                
                
                
    
        #g_sel.vs['color'] = vertex_col
        
        #view = [0.0,0.0]
        
        #Z_list_all_modules_file = os.path.abspath("module_" + str(mod_index) + "_from_left.eps")

        #layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])
        
        #g_sel['layout'] = layout2D.tolist()
    
        ##ig.plot(g_sel, Z_list_all_modules_file, vertex_size = 0.5,    edge_width = 0.1, edge_curved = False)
        #ig.plot(g_sel, Z_list_all_modules_file, vertex_size = np.array(vertex_size),    edge_width = 1, edge_curved = False,vertex_shape = vertex_shape)
        
        #Z_list_all_modules_files.append(Z_list_all_modules_file)
        
    #return Z_list_all_modules_files
    
#def plot_3D_igraph_all_modules_coomatrix_rel_coords_node_roles(community_vect,node_rel_coords,coomatrix,node_roles):
    
    
    #if (community_vect.shape[0] != node_rel_coords.shape[0]):
        #print "Warning, community_vect {} != node_coords {}".format(community_vect.shape[0], node_coords.shape[0])
        
    #if (community_vect.shape[0] != node_roles.shape[0]):
        #print "Warning, community_vect {} != node_roles {}".format(community_vect.shape[0], node_roles.shape[0])
        

    #print community_vect.shape
    #print node_rel_coords.shape
    
    #print coomatrix.shape
    
    ############ threshoding the number of dictictly displayed modules with the number of igraph colors
    
    #community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
    
    ############ extract edge list (with coords belonging to )
    
    #print np.unique(community_vect)
    
    #edge_col_inter = []
    #edge_list_inter = []
    #edge_weights_inter = []
    
    #edge_col_intra = []
    #edge_list_intra = []
    #edge_weights_intra = []
    
    #for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
        
        #if (community_vect[u] == community_vect[v]):
            
            #edge_list_intra.append((u,v))
            #edge_weights_intra.append(w)
            #edge_col_intra.append(igraph_colors[community_vect[u]])
        #else:
            
            #edge_list_inter.append((u,v))
            #edge_weights_inter.append(w)
            #edge_col_inter.append("lightgrey")
            
            
    #edge_list = edge_list_inter + edge_list_intra
    
    #edge_weights = edge_weights_inter + edge_weights_intra 
    
    #edge_col = edge_col_inter + edge_col_intra
    
    
    
    
    ##edge_col_intra = []
    ##edge_list_intra = []
    ##edge_weights_intra = []
    
    ##for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
        
        ##if (community_vect[u] == community_vect[v]):
            
            ##edge_list_intra.append((u,v))
            ##edge_weights_intra.append(w)
            ##edge_col_intra.append(igraph_colors[community_vect[u]])
            
    ######################################################## All modules 
    
    ##edge_list = edge_list_intra
    
    ##edge_weights = edge_weights_intra 
    
    ##edge_col = edge_col_intra
    
    
        
        
        
        
    #g_all= ig.Graph(edge_list)
    
    #g_all.es["weight"] = edge_weights
    
    #g_all.es['color'] = edge_col
    ##print g_all
    
    
    
    
    
    ########## colors (module belonging) and shape - size (roles)
    
    #vertex_col = []
    
    #vertex_shape = []
    
    #vertex_size = []
    
    #print len(g_all.vs),node_roles.shape[0]
    
    #for i,v in enumerate(g_all.vs):
        
        #### color
        #mod_index = community_vect[i]
        #if (mod_index != len(igraph_colors)-1):
            #vertex_col.append(igraph_colors[mod_index])
        #else:
            #vertex_col.append("lightgrey")
            
        #### node size
        #if node_roles[i,0] == 1:
            #vertex_size.append(5.0)
        #elif node_roles[i,0] == 2:
            #vertex_size.append(10.0)
            
        #### shape
        #if node_roles[i,1] == 1:
            #vertex_shape.append("circle")
            
        #elif node_roles[i,1] == 2 or node_roles[i,1] == 5:
            
            #vertex_shape.append("rectangle")
    
        #elif node_roles[i,1] ==  3 or node_roles[i,1] == 6:
            
            #vertex_shape.append("triangle-up")
            
        #elif node_roles[i,1] == 4 or node_roles[i,1] == 7:
            #vertex_shape.append("triangle-down")
            
    
    #g_all.vs['color'] = vertex_col
    
    #views = [[0.0,0.0],[0.,90.0],[90.,0.0],[0.,-90.0]]
    
    #suf = ["_from_left","_from_front","_from_top","_from_behind"]
    
    #Z_list_all_modules_files = []
    
    #for i,view in enumerate(views):
    
        #print view
        
        #Z_list_all_modules_file = os.path.abspath("All_modules" + suf[i] + ".eps")

        #Z_list_all_modules_files.append(Z_list_all_modules_file)
        
        #layout2D = project2D_np(node_rel_coords, angle_alpha = view[0],angle_beta = view[1])
        
        #g_all['layout'] = layout2D.tolist()
    
        #ig.plot(g_all, Z_list_all_modules_file, vertex_size = np.array(vertex_size),    edge_width = 1, edge_curved = False,vertex_shape = vertex_shape)
        
    #return Z_list_all_modules_files
    