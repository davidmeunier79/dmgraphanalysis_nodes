# -*- coding: utf-8 -*-
"""
Support function for igraph
"""

import numpy as np
import os


import igraph as ig

import cairo as ca

from utils_dtype_coord import where_in_coords,find_index_in_coords
import math

nb_igraph_colors = 100

######################################## generate colors #################################

import matplotlib.cm as cm

def frac_tohex_tuple(val_rgb):
    
    return frac_tohex(val_rgb[0],val_rgb[1],val_rgb[2])
    
def frac_tohex(r,g,b):
    
    return int_tohex(r*255,g*255,b*255)
    
    
def int_tohex(r,g,b):
    
    hexchars = "0123456789ABCDEF"
    return "#" + hexchars[int(r / 16)] + hexchars[int(r % 16)] + hexchars[int(g / 16)] + hexchars[int(g % 16)] + hexchars[int(b / 16)] + hexchars[int(b % 16)]
     
     
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def hex_to_fracrgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(float(int(value[i:i+lv/3], 16))/255.0 for i in range(0, lv, lv/3))

def hex_to_fracrgba(value,alpha = 1.0):
    value = value.lstrip('#')
    lv = len(value)
    return tuple([float(int(value[i:i+lv/3], 16))/255.0 for i in range(0, lv, lv/3)] + [alpha])

def generate_RGB_colors(nb_colors):

    import colorsys
    import matplotlib.cm as cm
    
    N = 1000
    
    RGB_tuples = cm.get_cmap('rainbow',nb_colors)
    
    RGB_colors = []
    
    increment = 1
    val = 0
    
    
    while len(RGB_colors) < nb_colors:
        
        color = RGB_tuples(val)
        
        if not color in RGB_colors:
            RGB_colors.append(color)
            
        val = val+ 1.0/(float(increment))
        
        if (val > 1.0):
            
            increment = increment +1
            
            val = 1.0/(float(increment))
            
    return RGB_colors

    ### ok RGB (couleur pas assez differentes)
def generate_igraph_colors(nb_colors):

    import colorsys
    import matplotlib.cm as cm
    
    N = 1000
    
    RGB_tuples = cm.get_cmap('rainbow',N)
    #RGB_tuples = cm.get_cmap('spectral',N)
    
    print RGB_tuples
       
    
    #RGB_tuples = my_cmap[:N]
    
    #print RGB_tuples
    
    
    #print RGB_tuples
    
    igraph_colors = []
    
    increment = 1
    
    val = 0.0
    
    list_val = []
    
    #while len(igraph_colors) < nb_colors:
        
        ##print val
        ##print RGB_tuples[int(val*N)]
        
        
        #color = frac_tohex_tuple(RGB_tuples(val))
        
        ##print color
        
        #if not color in igraph_colors:
            #igraph_colors.append(color)
            
        
        #val = val+ 1.0/increment
        
        #if (val > 1.0):
            
            #increment = increment +1
            
            #val = 1.0/increment
            
            
        ##print len(igraph_colors)
    
    while len(igraph_colors) < nb_colors:
        
        #print val, val*N
        #print RGB_tuples[int(val*N)]
        
        color = frac_tohex_tuple(RGB_tuples(val))
        
        #print color
        
        if not color in igraph_colors:
            igraph_colors.append(color)
            
        val = val+ 1.0/(3.0*float(increment))
        
        if (val > 1.0):
            
            increment = increment +1
            
            val = 1.0/(3.0*float(increment))
            
            
        #print len(igraph_colors)
    
    
    return igraph_colors

    #OK marche HSV
#def generate_igraph_colors(nb_colors):

    #import colorsys
    
    #N = 1000
    #HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    #RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    ##print RGB_tuples
    
    #igraph_colors = []
    
    #increment = 1
    #val = 0
    
    
    #while len(igraph_colors) < nb_colors:
        
        ##print val, val*N
        ##print RGB_tuples[int(val*N)]
        
        #color = frac_tohex_tuple(RGB_tuples[int(val*N)])
        
        ##print color
        
        #if not color in igraph_colors:
            #igraph_colors.append(color)
            
        #val = val+ 1.0/(3.0*float(increment))
        
        #if (val >= 1.0):
            
            #val = 1.0/(3.0*float(increment))
            
            #increment = increment +1
            
        ##print len(igraph_colors)
    
    
    #return igraph_colors

igraph_colors = generate_igraph_colors(nb_igraph_colors)

RGB_colors = generate_RGB_colors(nb_igraph_colors)
#igraph_colors = ['red','blue','green','yellow','brown','purple','orange','black']


def add_vertex_colors(g_all,community_vect,list_colors = igraph_colors):
    
    vertex_col = []
    vertex_label_col = []
    
    for i,v in enumerate(g_all.vs):
        mod_index = community_vect[i]
        if (mod_index != len(list_colors)-1):
            vertex_col.append(list_colors[mod_index])
            vertex_label_col.append(list_colors[mod_index])
        else:
            vertex_col.append("lightgrey")
            vertex_label_col.append(list_colors[mod_index])
    
    g_all.vs['color'] = vertex_col
    g_all.vs['label_color'] = vertex_label_col
    
def  create_module_edge_list(coomatrix,community_vect,list_colors = igraph_colors):
    
    ########### threshoding the number of dictictly displayed modules with the number of igraph colors
    
    community_vect[community_vect > len(igraph_colors)-1] = len(igraph_colors)-1
    
    ########### extract edge list (with coords belonging to )
    
    print np.unique(community_vect)
    
    edge_col_inter = []
    edge_list_inter = []
    edge_weights_inter = []
    
    edge_col_intra = []
    edge_list_intra = []
    edge_weights_intra = []
    
    for u,v,w in zip(coomatrix.row,coomatrix.col,coomatrix.data):
        
        if (community_vect[u] == community_vect[v]):
            
            edge_list_intra.append((u,v))
            edge_weights_intra.append(w)
            edge_col_intra.append(igraph_colors[community_vect[u]])
        else:
            
            edge_list_inter.append((u,v))
            edge_weights_inter.append(w)
            edge_col_inter.append("lightgrey")
            
            
    edge_list = edge_list_inter + edge_list_intra
    
    edge_weights = edge_weights_inter + edge_weights_intra 
    
    edge_col = edge_col_inter + edge_col_intra
    
    
    g_all= ig.Graph(edge_list)
    
    g_all.es["weight"] = edge_weights
    
    g_all.es['color'] = edge_col
    #print g_all
    
    return g_all
    
######################################## igraph 3D #######################################
     
def project2D_np(node_coords, angle_alpha = 0.0, angle_beta = 0.0):

    #node_coords = np.transpose(np.vstack((node_coords[:,1],-node_coords[:,2]*0.5,node_coords[:,0])))
    node_coords = np.transpose(np.vstack((node_coords[:,1],-node_coords[:,2],node_coords[:,0])))
    
    #print node_coords
    
    ##0/0
    
    angle_alpha = angle_alpha + 10.0
    angle_beta = angle_beta + 5.0

    print node_coords.shape
    
    #layout2D = project2D(node_coords.tolist(),0,0)
    layout2D = project2D(node_coords.tolist(),np.pi/180*angle_alpha,np.pi/180*angle_beta)
    
    #node_coords = np.transpose(np.vstack((node_coords[:,1],node_coords[:,2],node_coords[:,0])))
    
    #print node_coords.shape
    
    ##layout2D = project2D(node_coords.tolist(),0,0)
    #layout2D = project2D(node_coords.tolist(),0,0)
    
    return layout2D
    
def project2D(layout, alpha, beta):
    '''
    This method will project a set of points in 3D to 2D based on the given
    angles alpha and beta.
    '''
    # Calculate the rotation matrices based on the given angles.
    c = np.matrix([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
    c = c * np.matrix([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    b = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Hit the layout, rotate, and kill a dimension
    layout = np.matrix(layout)
    #x,y,z = (b * (c * layout.transpose())).transpose()
    
    X = (b * (c * layout.transpose())).transpose()
    
    #print X.shape
    
    proj = [[X[i,0],X[i,1],X[i,2]] for i in range(X.shape[0])]
     
    #print proj
    
    x,y,z = zip(*proj)
    
    #graph.vs['x2'], graph.vs['y2'], graph.vs['z2'] = zip(*layout2D)
    minX, maxX = min(x), max(x)
    minY, maxY = min(y), max(y)
    #minZ, maxZ = min(z), max(z)
    
    
    print minX, maxX
    print minY, maxY
        
    
    layout2D_x = (x - minX) / (maxX - minX)
    layout2D_y = (y - minY) / (maxY - minY)
    
    print layout2D_x
    print layout2D_y
        
    layout2D = np.transpose(np.vstack((layout2D_x,layout2D_y)))
    
    #print layout2D.shape
    
    return layout2D
    
################################################################# Methode to fill Graph properties ################################################################################
    
def return_base_weighted_graph(int_matrix):

    mod_list = int_matrix.tolist()
    
    #print mod_list
    
    g= ig.Graph.Weighted_Adjacency(mod_list,mode=ig.ADJ_MAX)
    
    return g
    
def add_non_null_labels(g, labels = []):
    
    null_degree_index, = np.where(np.array(g.degree()) == 0)
    
    #print null_degree_index
    
    np_labels = np.array(labels,dtype = 'string')
    
    #print np_labels
    
    np_labels[null_degree_index] = ""
    
    #print np_labels
        
    if len(labels) == len(g.vs):
    
        g.vs['label'] = np_labels.tolist()
        
        g.vs['label_size'] = 15
    
        g.vs['label_dist'] = 2
        
        print g.vs['label']
    
#def  add_edge_colors(g,color_dict)
def add_node_shapes(g_all,node_roles):

    vertex_shape = []
    
    vertex_size = []
    
            
    for i,v in enumerate(g_all.vs):
    
        ### node size
        if node_roles[i,0] == 1:
            #vertex_size.append(5.0)
            v["size"] = 5.0
        elif node_roles[i,0] == 2:
            #vertex_size.append(10.0)
            v["size"] = 10.0
            
        ### shape
        if node_roles[i,1] == 1:
            v["shape"] = "circle"
            #vertex_shape.append("circle")
            
        elif node_roles[i,1] == 2 or node_roles[i,1] == 5:
            v["shape"] = "rectangle"
            #vertex_shape.append("rectangle")
    
        elif node_roles[i,1] ==  3 or node_roles[i,1] == 6:
            v["shape"] = "triangle-up"
            #vertex_shape.append("triangle-up")
            
        elif node_roles[i,1] == 4 or node_roles[i,1] == 7:
            v["shape"] = "triangle-down"
            #vertex_shape.append("triangle-down")
            
    #g_all.vs["size"] = np.array(vertex_size),   
    #g_all.vs["shape"] = vertex_shape
    