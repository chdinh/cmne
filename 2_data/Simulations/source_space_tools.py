# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:14:30 2018

@author: ju357
"""

import numpy as np
import mne
#import mayavi
#from mayavi import mlab
import plyfile
import plotly
from plyfile import PlyData, PlyElement
import random
import matplotlib as plt


def print_ply(fname, src, scals):
    """Takes a source object and scalar vertex values, transforms them to face values
    if face_colors is True, and saves them as a colored PLV file that can be opened with a software like Meshlab.
    src is the source space object, scals is the vertex scalar map and fname is the name
    of the ply output file, saved in ./ply_files folder"""
    
    #Color faces instead of vertices - associate each face with scalar average of surrounding verts.
    scal_face = np.zeros(src['tris'].shape[0])
    for c,x in enumerate(scal_face):
        scal_face[c] = np.mean(scals[src['tris'][c]])
    max_val = np.max(scal_face[~np.isnan(scal_face)])
    max_val_verts = np.max(scals)
    
    #Convert scalars to color map
    
    array_list = []
    for c,x in enumerate(src['tris']):
        
        color_val = scal_face[c]/max_val
        color = [int(np.ceil(256.0*val))-1 for val in plt.cm.YlGnBu(color_val)]
        data_tup = ([x[0], x[1], x[2]], color[0], color[1], color[2])
        array_list.append(data_tup)    
    
    faces_colored = np.array(array_list, dtype=[('vertex_indices', 'int32', (3,)),
                                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    #Make a list of tuples out of the vertex arrays. Multiply vertex positions
    # by 1000 to turn it in mm for graphics software like Blender

    rr_list = []
    for c,x in enumerate(src['rr']):
        
        color_val = scals[c]/max_val_verts
        color = [int(np.ceil(256.0*val))-1 for val in plt.cm.YlGnBu(color_val)]
        temp_tuple = (x[0]*1000, x[1]*1000, x[2]*1000, color[0], color[1], color[2])            
        rr_list.append(temp_tuple)
    
    vertex = np.array(rr_list,dtype=[('x', 'float64'), ('y', 'float64'), ('z', 'float64'),
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

#    vertex = np.array(rr_list,dtype=[('x', 'float64'), ('y', 'float64'), ('z', 'float64'), \
#                                    ('nx', 'float64'), ('ny', 'float64'), ('nz', 'float64')])
    
    el_vert = PlyElement.describe(vertex,'vertex')
    el_face = PlyElement.describe(faces_colored,'face')

    PlyData([el_vert, el_face], text=True).write('./ply_files/'+fname)
    
    return
    
    

def calculate_area(rr,tris):
    """Takes rr - an array of position of vertices and tris - indices of vertices that deliniates
    triangle face and returns the total area of the tesselation surface."""
    
    area = 0.0
    for row in tris:
        v1=rr[row[1],:]-rr[row[0],:]
        v2=rr[row[2],:]-rr[row[0],:]
        nml = np.cross(v1,v2)
        area = area + np.linalg.norm(nml)/2.0            

    print('Total surface area: ' + np.str(area))
    
    return area



def calculate_normals(rr,tris,solid_angle_calc=False,obs_point=np.zeros(3)):
    """Takes rr - an array of position of vertices and tris - indices of vertices that deliniates
    triangle face and returns vertex normals based on an (unweighted) average of neighboring face normals."""
    A = []
    area_list = []
    area = 0.0
    count=0
    nan_vertices = []
    for x in range(len(rr)):
        A.append([])
    solid_angle = 0
    for row in tris:
        v1=rr[row[1],:]-rr[row[0],:]
        v2=rr[row[2],:]-rr[row[0],:]
        nml = np.cross(v1,v2)
        area = area + np.linalg.norm(nml)/2.0
        area_list.append(area)
        nn_fc = nml/np.linalg.norm(nml)
        A[row[0]].append(nn_fc)
        A[row[1]].append(nn_fc)
        A[row[2]].append(nn_fc)
        if solid_angle_calc==True:
            R1 = rr[row[0]] - obs_point
            R2 = rr[row[1]] - obs_point
            R3 = rr[row[2]] - obs_point
            
            solid_angle = solid_angle + 2*np.arctan(np.dot(R1,np.cross(R2,R3))/ \
                (np.linalg.norm(R1)*np.linalg.norm(R2)*np.linalg.norm(R3) + np.dot(R1,R2)*np.linalg.norm(R3) + \
                np.dot(R1,R3)*np.linalg.norm(R2) + np.dot(R2,R3)*np.linalg.norm(R1)))  
    
    if solid_angle_calc==True:
        print('solid_angle at the point of observation estimated to:')
        print(solid_angle)    
        
    nn = np.zeros((len(rr),3))
    for c, ele in enumerate(A):
        vert_norm = np.zeros(3)
        for vec in ele:
            vert_norm = vert_norm + vec
        vert_norm = vert_norm/np.linalg.norm(vert_norm)
        nn[c,:]=vert_norm
        
    for c, ele in enumerate(A):
        if np.isnan(nn[c,:]).any(): #np.linalg.norm(nn[c,:]) == 0:
            neighbor_rows = np.where(tris==c)[0]
            neighbors = np.unique(tris[neighbor_rows])
            neighbors = neighbors[np.where(neighbors!=c)]
            normal = np.mean(nn[neighbors,:],axis=0)
            nn[c,:] = normal/np.linalg.norm(normal)
            count = count+1
            
            if np.isnan(nn[c,:]).any():
                nan_vertices.append(c)
            

    print('number of nan normals that have been smoothed = ' + str(count))
    print('Remaining NAN normals = ' + str(len(nan_vertices)))                
    print('Total surface area: ' + np.str(area))
    
    return (nn,area,area_list,nan_vertices)



def plot_surface(src):
        
    for x in src:
        
        vertices = x['rr']
        faces = x['tris']
        normals = x['nn']
            
        white = (1.0, 1.0, 1.0)  # RGB values for a white color
        gray = (0.5, 0.5, 0.5)  # RGB values for a gray color
        red = (1.0, 0.0, 0.0)  # RGB valued for a red color
        
        # Plot the cerebellum with normals
        mayavi.mlab.triangular_mesh(vertices[:, 0]*1000, vertices[:, 1]*1000, vertices[:, 2]*1000, faces, color=gray)    
        
        mayavi.mlab.quiver3d(vertices[:, 0]*1000, vertices[:, 1]*1000, vertices[:, 2]*1000,
                      normals[:, 0], normals[:, 1], normals[:, 2],
                      scale_factor=2)
    return



def add_ply_tesselation_to_source_space(src_orig, ply_filepath):
    """ Takes a PLY file, transforms it into a source space object and adds it to an existing source space (src_orig).
    Input:  ply_filepath = path (string) to ply file
    Output: src = source space with ply surface in the first and only element"""
    
    plydata_marty = PlyData.read(ply_filepath)
    rr = np.array([list(dat)[:3] for dat in plydata_marty['vertex']._data])/1000
    fc = np.array([dat for dat in plydata_marty['face'].data['vertex_indices']])
    src = src_orig.copy()[1]
    src['inuse'] = np.ones(len(rr)).astype(int)
    src['nn'] = np.concatenate((np.zeros((len(rr),2)),np.ones((len(rr),1))),axis=1)
    src['np'] = len(rr)
    src['ntri'] = len(fc)
    src['nuse'] = len(rr)
    src['nuse_tri'] = len(fc)
    src['rr'] = rr
    src['tris'] = fc
    src['use_tris'] = fc
    src['vertno'] = np.nonzero(src['inuse'])[0]
    src_out = src_orig.copy()
    src_out[0] = join_source_spaces(src_orig)
    src_out[1] = src
    
    return src_out



    
    
def join_source_spaces(src_orig):
    if len(src_orig)!=2:
        raise ValueError('Input must be two source spaces')
        
    src_joined=src_orig.copy()
    src_joined=src_joined[0]
    src_joined['inuse'] = np.concatenate((src_orig[0]['inuse'],src_orig[1]['inuse']))
    src_joined['nn'] = np.concatenate((src_orig[0]['nn'],src_orig[1]['nn']),axis=0)
    src_joined['np'] = src_orig[0]['np'] + src_orig[1]['np']
    src_joined['ntri'] = src_orig[0]['ntri'] + src_orig[1]['ntri']
    src_joined['nuse'] = src_orig[0]['nuse'] + src_orig[1]['nuse']
    src_joined['nuse_tri'] = src_orig[0]['nuse_tri'] + src_orig[1]['nuse_tri']
    src_joined['rr'] = np.concatenate((src_orig[0]['rr'],src_orig[1]['rr']),axis=0)
    triangles_0 = len(src_orig[0]['rr'])
    src_joined['tris'] = np.concatenate((src_orig[0]['tris'],src_orig[1]['tris']+triangles_0),axis=0)
    src_joined['use_tris'] = np.concatenate((src_orig[0]['use_tris'],src_orig[1]['use_tris']+triangles_0),axis=0)
    src_joined['vertno'] = np.nonzero(src_joined['inuse'])[0]

    return src_joined    
    
    
    
    
    
def sensitivity_map(fwd,subjects_dir,subject,plot=False):
    
    src_joined = join_source_spaces(src_orig=fwd['src'])
    vert_pos = src_joined['rr']
    stc_data = 1
    stc_d = np.tile(stc_data,(len(src_joined['vertno']),1))
    stc_all = mne.SourceEstimate(stc_d, vertices=[fwd['src'][0]['vertno'],fwd['src'][1]['vertno']], tmin=0, tstep=1, subject='fsaverage')
    picks_grad = mne.pick_types(fwd['info'], meg='grad', exclude = 'bads')
    picks_mag = mne.pick_types(fwd['info'], meg='mag', exclude = 'bads')
    picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True, exclude = 'bads')
    all_tris = src_joined['tris']
    
    G_grad = fwd['sol']['data'][picks_grad,:]
    G_mag = fwd['sol']['data'][picks_mag,:]
    G_eeg = fwd['sol']['data'][picks_eeg,:]
    
    stc_eeg = stc_all.copy()
    stc_grad = stc_all.copy()
    stc_mag = stc_all.copy()
    mag=np.zeros(src_joined['nuse'])
    
    for k in range(src_joined['nuse']):
        stc_eeg.data[k] = np.linalg.norm(G_eeg[:,k])
        stc_grad.data[k] = np.linalg.norm(G_grad[:,k])
        stc_mag.data[k] = np.linalg.norm(G_mag[:,k])
        mag[k]=np.linalg.norm(G_mag[:,k])
    
    mag_scalars = np.zeros(len(vert_pos[:,0]))
    mag_scalars[src_joined['vertno']] = mag
    
    if plot==True:    
        triangular_mesh(vert_pos[:,0], vert_pos[:,1], vert_pos[:,2], all_tris, scalars=mag_scalars)
        colorbar()
        
    return (stc_mag,stc_grad,stc_eeg)
    
    
    
def find_verts(center, radius, src, plot_patch = False):
    """Function to find all vertices within radius (in mm) of center vertex over uniform tesselation grid.
    src is supposed to be a source object (i.e., not a list of source objects)"""
    
    keys = src['vertno']
    values = range(0,len(src['vertno']))
    my_dic = dict(zip(keys,values))
    rr = src['rr']
        
    start_rr = center
    start_f = np.where(start_rr==src['tris'])[0][0]
    while np.where(src['tris'] == start_rr)[0].size==0:
        start_rr = random.choice(src['vertno'])
        
    current_face = start_f
    all_fc =  np.array([current_face])
    verts_new = np.array([start_rr])
    area = 0.0
    area_old = 1.0   

    used = np.array(range(src['np']))
    used[verts_new] = 0
    used_tri = np.array(range(src['ntri']))
    used_tri[all_fc] = 0

    r = radius
    wanted_area = np.pi*r**2*10**-6
    
    while area < wanted_area:
        if area_old == area:
            break
        #find new faces from the new verts
        for x in verts_new:
            neighbors = np.where(src['tris'] == x)[0]
            all_fc = np.concatenate((all_fc,neighbors),axis=0)
        
        all_fc = np.unique(all_fc)
        tris_new = all_fc[np.nonzero(used_tri[all_fc])]
        used_tri[all_fc] = 0
        
        #Calculate area of expanded area    
        
        area_old = area
        for faces in tris_new:
            v1=rr[src['tris'][faces][1],:]-rr[src['tris'][faces][0],:]
            v2=rr[src['tris'][faces][2],:]-rr[src['tris'][faces][0],:]
            nml = np.cross(v1,v2)
            area = area + np.linalg.norm(nml)/2.0

        #find vertices of face                    
        verts = src['tris'][all_fc]
        verts_new = verts[np.nonzero(used[verts])]
        used[verts_new] = 0
            
    verts = np.unique(src['tris'][all_fc])
#    mask = np.isin(verts,src['vertno'])
    mask = np.array([item in src['vertno'] for item in verts])
    verts = verts[mask]
    verts = np.array([my_dic[x] for x in verts.tolist()])

#            if np.isnan(cancellation_index):
#                raise ValueError('NaN cancellation value detected. Breaking.')

    if plot_patch == True:
        #Plot activated patch
        scalars = np.zeros((len(rr[:,0])))
        verts = np.unique(src['tris'][all_fc])
        scalars[verts] = 0.5
        start_v = src['tris'][start_f,:]
        scalars[start_v] = 1.0
        mayavi.mlab.figure()
        mayavi.mlab.triangular_mesh(rr[:,0], rr[:,1], rr[:,2], src['tris'], scalars=scalars,opacity=1.0)
        mayavi.mlab.colorbar()        

    return verts
    
    

    
    
    
    
    
    
      