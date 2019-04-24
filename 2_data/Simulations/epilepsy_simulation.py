
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_evoked
import os
from os import path as op
import sys
#from source_space_tools import print_ply

sys.path.insert(0, '/cluster/fusion/john/anaconda/lib/python2.7/site-packages/')

def simulate_ictal():

	#%%
	#Load sample subject data and labels from local labels folder

	data_path = sample.data_path()
	subjects_dir = op.join(data_path, 'subjects')
	raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
	proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg_proj.fif')
	raw.info['projs'] += proj
	raw.info['bads'] = ['MEG 2443', 'EEG 053']

	fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
	ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
	cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
	trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
	bem_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')

	fwd = mne.read_forward_solution(fwd_fname)
	fwd = mne.convert_forward_solution(fwd,surf_ori=True,force_fixed=True)
	fwd = mne.pick_types_forward(fwd, meg=True, eeg=False, exclude=raw.info['bads'])
	cov = mne.read_cov(cov_fname)
	info = mne.io.read_info(ave_fname)

	fs = raw.info['sfreq']

	def epileptiform(t, prestim, poststim, amp):
		fs = raw.info['sfreq']
		f = 2/t[-1]
		signal = np.concatenate((np.zeros(np.round(prestim*fs).astype(int)), 
				amp*np.exp(-10.0*t)*np.cos(t*f*2*np.pi),np.zeros(np.round(poststim*fs).astype(int))))[0:900]
		dt = fs**-1
		t_stimulus = np.concatenate((np.linspace(0,prestim-dt,fs*prestim),t+prestim,np.linspace(dt,poststim,fs*poststim)+t[-1]+prestim))[0:900]
		return t_stimulus,signal


	duration = 0.2
	t = np.linspace(0,duration,duration*fs)
	n_verts = [53109,69425,82958,95007]
	closest_verts = [np.argmin(np.abs(x-fwd['src'][0]['vertno'])) for x in n_verts]
	signal = np.zeros((305,900))

	for c,dip in enumerate(closest_verts):
		prestim = 0.5+c*0.2
		poststim = 0.8-c*0.2
		sensor_topography = fwd['sol']['data'][:,dip]
		time,trace = epileptiform(t=t, prestim=prestim, poststim=poststim, amp=10**-6)
		signal = signal+np.dot(sensor_topography.reshape(305,1),trace.reshape(1,900))

	noise = mne.io.read_raw_fif(data_path + '/MEG/sample/ernoise_raw.fif',preload=True)
	noise.filter(l_freq=1.0,h_freq=70.0)
	noise.pick_types(meg=True)
	noise_data = noise.get_data()
	start_rand = ((noise_data.shape[1]-len(time))*np.random.rand(1)).astype(int)[0]

	#sim_data is the output
	sim_data = signal + noise_data[:,start_rand:start_rand+len(time)]

#	plt.figure()
#	plt.plot(time,sim_data[0,:])
#	plt.show()

	#print to view the activated vertices (can be opened in blender)
#	scals = np.zeros(fwd['src'][0]['np'])
#	scals[n_verts] = 1.0
#	print_ply(fname='dipole_1.ply', src=fwd['src'][0], scals=scals)


	return sim_data

print(simulate_ictal())


