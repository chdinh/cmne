#**
# @file     1_run_hyper_evaluation.py
# @author   Christoph Dinh <christoph.dinh@mne-cpp.org>
# @version  1.0
# @date     April, 2019
#
# @section  LICENSE
#
# Copyright (C) 2019, Christoph Dinh. All rights reserved.
#
# @brief    Run hyper evaluation
#**

#%% Imports
import sys
sys.path.append("./I_cmne/II_training") #Add relative path to include modules

from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData

import config as cfg

from train_LSTM import train_LSTM

#%% Settings
# 0 - Sample Data
#settings = CMNESettings(repo_path='D:/Git/cmne/', data_path='D:/Git/mne-cpp/bin/MNE-sample-data/',
#                       fname_raw='sample_audvis_filt-0-40_raw.fif',
#                       fname_inv='sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif',
#                       fname_eve='sample_audvis_filt-0-40_raw-eve.fif',
#                       fname_test_idcs='sample_audvis-test-idcs.txt')
# 1 - Local
#settings = CMNESettings(repo_path='D:/Users/Christoph/Git/cmne/', data_path='D:/Data/MEG/jgs/170505/processed/')
settings = CMNESettings(repo_path=cfg.repo_path, data_path=cfg.data_path,
                        fname_raw=cfg.fname_raw,
                        fname_inv=cfg.fname_inv,
                        fname_eve=cfg.fname_eve,
                        fname_test_idcs=cfg.fname_test_idcs
                       )


#%% Data
event_id, tmin, tmax = 1, -0.2, 0.5

data = CMNEData(cmne_settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax)


#%% train
train_LSTM(settings, data)
