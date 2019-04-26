#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:43:47 2019

@author: ju357
"""

import mne
import matplotlib.pylab as plt
import pyimpress as pyi
import numpy as np
from mne.minimum_norm import apply_inverse
import os.path as op
plt.ion()
subjects_dir = '/autofs/cluster/fusion/data/FreeSurfer'
subject = 'jgs-20160519'
protocol_path = '/space/megraid/research/MEG/tal/subj_hst563/170322/'
raw_fname = protocol_path + 'emo2_raw.fif'
trans = protocol_path + 'trans.fif'
bem_dir = op.join(subjects_dir, subject, 'bem')
fname_mri = op.join(subjects_dir, subject, 'mri', 'T1.mgz')


#/autofs/cluster/fusion/data/MEG_EEG/john/170322
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw =raw.pick_types(meg=True, eeg=False, eog=True, ecg=True, stim=True)
raw.info['bads'] +=  [u'MEG0111', u'MEG2413', u'MEG2412', u'MEG1643']
#mne.gui.coregistration(raw,subject=subject, subjects_dir=subjects_dir)

#pyi.utils.compute_ecg_eog_proj(raw)
raw.apply_proj()
raw.filter(.5,30,filter_length='auto',
           l_trans_bandwidth='auto',h_trans_bandwidth='auto')
event_id = dict(angry=1, houses=3)
tmin, tmax, baseline = -.2, 1,(-.2, 0)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=True)
events = pyi.utils.run_events(raw)[0]
reject=dict(grad=4000e-13, eog=350e-6, mag=4e-12)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=reject,
                    preload=True)
epochs.resample(150., npad='auto')
evokeds = {key:epochs[key].average() for key in event_id.keys()}
evokeds['angry'].plot_joint([.135,.152,.242,.352])
contrast = mne.combine_evoked([evokeds['angry'], - evokeds['houses']], weights='equal')
contrast.plot_joint([.135,.17,.2,.25])


cov = mne.compute_covariance(epochs, tmax=0)

##
#mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir)
conductivity = (0.3,)
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
src = mne.setup_source_space(subject, spacing='ico5',
                             subjects_dir=subjects_dir,
                             add_dist=False)
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem, meg=True, eeg=False, n_jobs=2)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2, depth=0.8)
vol_src = mne.setup_volume_source_space(
    subject, mri=fname_mri, pos=7.0, subjects_dir=subjects_dir, verbose=True)

fwd_vol =mne.make_forward_solution(raw_fname, trans, vol_src, bem=bem,
                            mindist=5.0,  # ignore sources<=5mm from innerskull
                            meg=True, eeg=False, n_jobs=1)
inv_vol = mne.minimum_norm.make_inverse_operator(raw.info, fwd_vol, cov)


# dip = mne.fit_dipole(evokeds['angry'], cov, bem, trans)[0]
#
# # Plot the result in 3D brain with the MRI image.
# dip.plot_locations(trans, subject, subjects_dir, mode='orthoview')


snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"
pick_ori = None


stc_faces, stc_houses = (apply_inverse(evokeds['angry'], inv, lambda2, method, pick_ori),
                                  apply_inverse(evokeds['houses'], inv, lambda2, method, pick_ori))

stc_faces.plot(time_viewer=True,hemi='split',
                      views=['lateral','medial'], surface='inflated',initial_time=.127,
         subject=subject, subjects_dir=subjects_dir)


stc = stc_faces - stc_houses
stc.plot(time_viewer=True,hemi='split',
                      views=['lateral','medial'], surface='inflated',initial_time=.215,
         subject=subject, subjects_dir=subjects_dir)

##
gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=12)
triggers = epochs.events[:, 2]
faces_vs_houses = (triggers[np.in1d(triggers, (1, 3))] == 3).astype(int)
gat.fit(epochs[('angry', 'houses')], y=faces_vs_houses)
gat.score(epochs[('angry', 'houses')], y=faces_vs_houses)
gat.plot(vmin=.6,vmax=.8,title='faces_vs_houses')
gat.plot_diagonal(title='faces_vs_houses')


dipoles, residual = mne.beamformer.rap_music(evokeds['l1'], fwd_vol, cov,
                              n_dipoles=2, return_residual=True)
