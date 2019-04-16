#!/usr/bin/env python
repo_path='D:/Users/Christoph/GitHub/bio/'
data_path='D:/Data/MEG/jgs/170505/processed/'

fname_raw='sample_audvis_filt-0-40_raw.fif',
fname_inv='sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif',
fname_eve='sample_audvis_filt-0-40_raw-eve.fif',
fname_test_idcs='sample_audvis-test-idcs.txt'



# 0 - Sample Data
#settings = BioSettings(repo_path='D:/GitHub/bio/', data_path='D:/GitHub/mne-cpp/bin/MNE-sample-data/',
#                       fname_raw='sample_audvis_filt-0-40_raw.fif',
#                       fname_inv='sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif',
#                       fname_eve='sample_audvis_filt-0-40_raw-eve.fif',
#                       fname_test_idcs='sample_audvis-test-idcs.txt')
# 1 - Azure Windows
#settings = BioSettings(repo_path='C:/Git/bio/', data_path='Z:/MEG/jgs/170505/processed/')
# 2 - Azure Linux
#settings = BioSettings(repo_path='/home/chdinh/Git/bio/', data_path='/cloud/datasets/MNE-sample-data/')
# 3 - Azure Windows Simulation
#settings = BioSettings(repo_path='C:/Git/bio/', data_path='Z:/Simulation/',
#                       fname_raw='SpikeSim2000_fs900_raw.fif',
#                       fname_inv='SpikeSim2000_fs900_raw-ico-4-meg-eeg-inv.fif',
#                       fname_eve='SpikeSim2000_fs900_raw-eve.fif',
#                       fname_test_idcs='SpikeSim2000_fs900_raw-test-idcs.txt')