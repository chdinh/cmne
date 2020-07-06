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
import os
import sys
sys.path.append('D:/Users/Christoph/Git/cmne/I_implementation/I_cmne/II_training') #Add relative path to include modules
sys.path.append('D:/Users/Christoph/Git/cmne/I_implementation/helpers')
sys.path.append('D:/Users/Christoph/Git/cmne/I_implementation/I_cmne/I_hyperparameter_evaluation')


from helpers.cmnesettings import CMNESettings
from helpers.cmnedata import CMNEData

import config as cfg
import random

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
train_percentage = 0.85
cross_validation_percentage = 0.85

data = CMNEData(cmne_settings=settings)
data.load_data(event_id=event_id, tmin=tmin, tmax=tmax, train_percentage=train_percentage)

num_train_idcs = len(data.train_idcs())

num_cross_iterations = 10
fname_cross_validation_idcs_prefix = 'assr_270LP_fs900_cross_idcs_it_'

# TBD load model across cross validations
for iteration in range(num_cross_iterations):
    fname_cross_validation_idcs = settings.data_path() + fname_cross_validation_idcs_prefix + str(iteration) + '.txt'

    if os.path.isfile(fname_cross_validation_idcs):
        cross_validation_train_idcs = []
        with open(fname_cross_validation_idcs, "r") as f:
            for line in f:
                cross_validation_train_idcs.append(int(line.strip()))
    else:
        #split train and test
        random.seed(42)
        cross_validation_train_idcs = random.sample(range(num_train_idcs), (int)(num_train_idcs*cross_validation_percentage))
        with open(fname_cross_validation_idcs, "w") as f:
            for idx in cross_validation_train_idcs:
                f.write(str(idx) +"\n")
    #%% train
    train_LSTM(settings, data, idx=cross_validation_train_idcs)
