#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Some random functions that are useful in M/EEG data simulations

@author: ju357
"""

import numpy as np
import source_space_tools

def neighbor_dictionary(src):
    """
    Takes as source space as input and returns a dictionary with every vertex as
    keys and its neighbors as values
    """
    
    keys = range(0,src['np'])
    values = [ [] for i in range(src['np']) ]
    my_dic = dict(zip(keys,values))
    
    for triangles in src['tris']:
        for vert in triangles:
            for neighbor in triangles:
                my_dic[vert].append(neighbor)
    
    for key,val in my_dic.items():
        val = list(set(val))
        my_dic[key] = val
        
    return my_dic


def blurring(x, src, smoothing_steps, spread = False):
    """
    Takes a vector source estimate (x) and source object as input and outputs a
    blurred estimate blurred_x. If smoothing_steps is 'cover', it will keep 
    smoothing until all nan elements of the cortex has been smoothed.
    """
    
    neighbor_dic = neighbor_dictionary(src)
    blurred_x = x.copy()
    verts = np.array(range(0,src['np']))
    c=0
    
    if spread:
        print('computing smoothing steps...')
        while(np.sum(np.isnan(blurred_x)) > 0):
            for vert_ind in verts[np.where(np.isnan(blurred_x))[0]]:
                neighbors = neighbor_dic[vert_ind]
                blurred_x[vert_ind] = np.nanmean(blurred_x[neighbors])
            c = c+1
        
    else:
        print('computing smoothing steps...')
        for c in range(0,smoothing_steps):
            print(c)
            for vert_ind,dipole in enumerate(x):
                neighbors = neighbor_dic[vert_ind]
                blurred_x[vert_ind] = np.nanmean(neighbors)
            c = c+1
                
    return blurred_x


def blurred_sensitivity_map(fwd, sensor_index, src_space_index, ply_fname, smoothing_steps, spread = False):
    """
    Returns the blurred source estimation and prints a sensitivity map to
    /ply_files/ply_fname.
    """
    
    src = fwd['src'][src_space_index]
    
    if src_space_index == 1:
        sensitivities = np.linalg.norm(fwd['sol']['data'][sensor_index, \
                                       fwd['src'][0]['nuse']:fwd['src'][0]['nuse']+ \
                                        fwd['src'][1]['nuse']],axis=0)
    else:
        sensitivities = np.linalg.norm(fwd['sol']['data'][sensor_index, \
                                       0:fwd['src'][0]['nuse']],axis=0)
    
    scalars = np.empty((src['np']))
    scalars[:] = np.nan
    scalars[src['inuse']==1] = sensitivities
    
    blurred_x = blurring(x=scalars, src=src, smoothing_steps=smoothing_steps, spread = spread)
    
    source_space_tools.print_ply(fname=ply_fname,src=fwd['src'][src_space_index],scals=blurred_x)
    
    print('finished printing sensitivity map to /ply_files/'+ply_fname)
    
    return blurred_x
    
    
    
    
    
    
    
    
    
    
    
    
    