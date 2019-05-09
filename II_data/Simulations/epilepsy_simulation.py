
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_evoked
import os
from os import path as op
import sys
from source_space_tools import print_ply

sys.path.insert(0, '/cluster/fusion/john/anaconda/lib/python2.7/site-packages/')

def simulate_ictal():

    #%%
    #Load sample subject data and labels from local labels folder

    data_path = '/autofs/cluster/fusion/chdinh/Shared_VM_Storage/Datasets/MEG/jgs/170505/processed/'
    raw = mne.io.read_raw_fif(data_path+'assr_270LP_fs900_raw.fif', preload=True)#preload to allow for data manipulation
    raw.filter(l_freq=1.0,h_freq=70.0)
    raw_data = raw.get_data()
    fwd_fname = data_path + 'assr_43_223_si_1_270LP_fs810_raw-ico-4-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.convert_forward_solution(fwd,surf_ori=True,force_fixed=True)
    fwd = mne.pick_types_forward(fwd, meg=True, eeg=True)
    info = raw.info
    fs = raw.info['sfreq']

    stim_duration = 1.5
    samples_duration = int(1.5*fs)
    duration = 0.2
    t = np.linspace(0,duration,duration*fs)
    n_verts = [71187,87750,107723,125181]
    closest_verts = [np.argmin(np.abs(x-fwd['src'][0]['vertno'])) for x in n_verts]
    number_of_epochs = 200
    time_samples = range(0,int(200*1.5*fs))
    events = mne.read_events(data_path+'assr_270LP_fs900_raw-eve.fif')
    noise = np.array([]).reshape(372,0)
    for i in range(0,600):
        add_on = np.zeros((372,int(0.5*fs)))
        add_on[0:366,:] = raw_data[0:366,events[i,0]+int(1.0*fs):events[i,0]+int(1.5*fs)]
        add_on[366:372,:] = raw_data[370:376,events[i,0]+int(1.0*fs):events[i,0]+int(1.5*fs)]
        noise = np.concatenate((noise,add_on),axis=1)
    
    def epileptiform(t, prestim, poststim, amp):
        f = 2/t[-1]
        signal = np.concatenate((np.zeros(np.round(prestim*fs).astype(int)), 
                amp*np.exp(-10.0*t)*np.cos(t*f*2*np.pi),np.zeros(np.round(poststim*fs).astype(int))))[0:samples_duration]
        dt = fs**-1
        t_stimulus = np.concatenate((np.linspace(0,prestim-dt,fs*prestim),t+prestim,np.linspace(dt,poststim,fs*poststim)+t[-1]+prestim))[0:samples_duration]
        return t_stimulus,signal
    
    def spike(closest_verts,fwd):
        signal = np.zeros((fwd['sol']['data'].shape[0],samples_duration))
        for c,dip in enumerate(closest_verts):
            prestim = 0.5+c*0.2
            poststim = 0.8-c*0.2
            sensor_topography = fwd['sol']['data'][:,dip]
            time,trace = epileptiform(t=t, prestim=prestim, poststim=poststim, amp=10**-6)
            signal = signal+np.dot(sensor_topography.reshape(fwd['sol']['data'].shape[0],1),trace.reshape(1,samples_duration))
    
        sim_data = signal
        
        return sim_data
    
    episode = spike(closest_verts,fwd)
    signal = np.repeat(episode,repeats=200,axis=1)
    data = noise+signal    
    activation_times = np.zeros(data.shape[1])
    activation_times[np.linspace(0,(number_of_epochs-1)*int(fs*1.5),number_of_epochs).astype(int)] = 1
    #saw sim_data in raw file. STI 001 is the trigger channel.
    raw_data_shape = raw_data.shape
    raw._data = np.zeros(raw_data_shape)
    raw._data[0:366,0:data.shape[1]] = data[0:366,:]
    raw._data[370:376,0:data.shape[1]] = data[366:372,:]
    raw._data[380,:] = np.zeros(raw._data[380,:].shape)
    raw._data[380,0:data.shape[1]] = activation_times

    events = mne.find_events(raw, stim_channel='STI101',
                     consecutive=False, verbose=True, initial_event=True)
                                             
    raw.save('./epilepsy_simulation_raw.fif',overwrite=True)
    mne.write_events('./epilepsy_simulation-eve.fif', events)
    
    #epochs = mne.Epochs(raw, events=events, event_id=[1], tmin=0, tmax=1.5, 
#     ...:                     preload=True, proj = False)#, add_eeg_ref=False) 
    

#    #print to view the activated vertices (can be opened in blender)
#    scals = np.zeros(fwd['src'][0]['np'])
#    scals[n_verts] = 1.0
#    print_ply(fname='dipole_1.ply', src=fwd['src'][0], scals=scals)


    return 
    
    
print(simulate_ictal())


